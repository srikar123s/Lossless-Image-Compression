# streamlit_app.py
# CAP-RC+++ Streamlit app: multi-file, per-image downloads, clean UI

import streamlit as st
import numpy as np
import io, os, struct, time, pandas as pd, warnings, contextlib, sys, tempfile, shutil
from PIL import Image
from bitarray import bitarray
from collections import Counter
import matplotlib.pyplot as plt

# note: JPEG-LS comparison removed from this build

warnings.filterwarnings("ignore")

# ---------------------------
# Helpers
# ---------------------------
@contextlib.contextmanager
def suppress_output():
    """Temporarily silence stdout/stderr (used during heavy loops)."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()

# ---------------------------
# Range coder (24-bit state)
# ---------------------------
FULL, HALF, QUARTER, THREE_QUARTER = 1 << 24, 1 << 23, 1 << 22, 3 << 22

class RangeEncoder:
    def __init__(self):
        self.low, self.high, self.pending, self.out_bits = 0, FULL - 1, 0, []

    def _emit_bit(self, b):
        self.out_bits.append(b)
        while self.pending > 0:
            self.out_bits.append(1 - b)
            self.pending -= 1

    def encode(self, sym, cum, total):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum[sym + 1]) // total - 1
        self.low  = self.low + (rng * cum[sym]) // total
        while True:
            if self.high < HALF:
                self._emit_bit(0); self.low <<= 1; self.high = (self.high << 1) | 1
            elif self.low >= HALF:
                self._emit_bit(1); self.low = (self.low - HALF) << 1; self.high = ((self.high - HALF) << 1) | 1
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low = (self.low - QUARTER) << 1
                self.high = ((self.high - QUARTER) << 1) | 1
            else:
                break

    def finish(self):
        self._emit_bit(0 if self.low < QUARTER else 1)

    def to_bytes(self):
        b = bitarray(self.out_bits)
        while len(b) % 8: b.append(0)
        return b.tobytes()

class RangeDecoder:
    def __init__(self, data):
        self.low, self.high, self.code = 0, FULL - 1, 0
        self.data = bitarray(endian='big'); self.data.frombytes(data)
        self.pos = 0
        for _ in range(24):
            self.code = (self.code << 1) | self._bit()
    def _bit(self):
        if self.pos >= len(self.data): return 0
        b = 1 if self.data[self.pos] else 0
        self.pos += 1
        return b
    def decode(self, cum, total):
        rng = self.high - self.low + 1
        val = ((self.code - self.low + 1) * total - 1) // rng
        sym = 0
        while cum[sym + 1] <= val: sym += 1
        self.high = self.low + (rng * cum[sym + 1]) // total - 1
        self.low  = self.low + (rng * cum[sym]) // total
        while True:
            if self.high < HALF: pass
            elif self.low >= HALF:
                self.code -= HALF; self.low -= HALF; self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.code -= QUARTER; self.low -= QUARTER; self.high -= QUARTER
            else: break
            self.low <<= 1; self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self._bit()
        return sym

# ---------------------------
# Predictor, context and color decorrelation (YCoCg)
# ---------------------------
SCALE = 256

def predict_pixel_weighted(x, y, img):
    A = int(img[y, x-1]) if x>0 else 0
    B = int(img[y-1, x]) if y>0 else 0
    C = int(img[y-1, x-1]) if (x>0 and y>0) else 0
    grad_h = abs(A - C); grad_v = abs(B - C)
    wa = SCALE // (1 + grad_v); wb = SCALE // (1 + grad_h)
    return (A*wa + B*wb)//(wa+wb) if (wa+wb) else (A+B)//2

def context_for_pixel(x, y, img):
    A = int(img[y, x-1]) if x>0 else 0
    B = int(img[y-1, x]) if y>0 else 0
    g = abs(A - B)
    return 0 if g < 4 else 1 if g < 16 else 2 if g < 64 else 3

def rgb_to_ycocg_uint16(img):
    R, G, B = img[...,0].astype(np.int32), img[...,1].astype(np.int32), img[...,2].astype(np.int32)
    Co = R - B
    tmp = B + (Co >> 1)
    Cg = G - tmp
    Y = tmp + (Cg >> 1)
    return np.stack([Y, Co + 512, Cg + 512], axis=2).astype(np.uint16)

def ycocg_to_rgb_uint8(ycocg):
    Y = ycocg[...,0].astype(np.int32)
    Co = ycocg[...,1].astype(np.int32) - 512
    Cg = ycocg[...,2].astype(np.int32) - 512
    tmp = Y - (Cg >> 1)
    G = Cg + tmp
    B = tmp - (Co >> 1)
    R = B + Co
    return np.clip(np.stack([R,G,B], axis=2), 0, 255).astype(np.uint8)

# ---------------------------
# Channel compress/decompress (contextual)
def compress_channel_contextual(img_channel, num_contexts=4, offset=2048):
    h, w = img_channel.shape
    ctx_res = [[] for _ in range(num_contexts)]
    for y in range(h):
        for x in range(w):
            pred = predict_pixel_weighted(x,y,img_channel)
            res = int(img_channel[y,x]) - pred
            ctx_res[context_for_pixel(x,y,img_channel)].append(res + offset)
    ctx_symbols, ctx_counts = [], []
    for lst in ctx_res:
        freq = Counter(lst)
        symbols = sorted(freq.keys())
        counts = [freq[s] for s in symbols]
        ctx_symbols.append(symbols); ctx_counts.append(counts)
    ctx_cum, ctx_totals = [], []
    for counts in ctx_counts:
        cum = [0]; [cum.append(cum[-1] + c) for c in counts]
        ctx_cum.append(cum); ctx_totals.append(cum[-1])
    enc = RangeEncoder()
    for y in range(h):
        for x in range(w):
            pred = predict_pixel_weighted(x,y,img_channel)
            res = int(img_channel[y,x]) - pred
            s = res + offset
            ctx = context_for_pixel(x,y,img_channel)
            idx = ctx_symbols[ctx].index(s)
            enc.encode(idx, ctx_cum[ctx], ctx_totals[ctx])
    enc.finish()
    return enc.to_bytes(), ctx_symbols, ctx_counts, offset, h, w

def decompress_channel_contextual(data, ctx_symbols, ctx_counts, offset, h, w):
    ctx_cum, ctx_totals = [], []
    for counts in ctx_counts:
        cum = [0]; [cum.append(cum[-1] + c) for c in counts]
        ctx_cum.append(cum); ctx_totals.append(cum[-1])
    dec = RangeDecoder(data)
    img = np.zeros((h,w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            ctx = context_for_pixel(x,y,img)
            if ctx_totals[ctx] == 0:
                val = 0
            else:
                idx = dec.decode(ctx_cum[ctx], ctx_totals[ctx])
                val = ctx_symbols[ctx][idx] - offset
            pred = predict_pixel_weighted(x,y,img)
            img[y,x] = int(np.clip(pred + val, 0, 65535))
    return img

# ---------------------------
# Main processing function
# ---------------------------
def process_images(files):
    if not files:
        return {
            "df": pd.DataFrame(),
            "visuals": [],
            "csv_bytes": b"",
            "zip_bytes": b"",
            "per_items": [],
            "charts": []
        }

    temp_dir = tempfile.mkdtemp(prefix="caprcppp_")
    rows, visuals, per_links = [], [], []

    for f in files:
        try:
            file_name = f.name
            file_bytes = f.getvalue()
            base, ext = os.path.splitext(os.path.basename(file_name))
            img = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
            h, w = img.shape[:2]
            orig_kb = len(file_bytes) / 1024.0

            # optional YCoCg decorrelation (we always use it here)
            proc = rgb_to_ycocg_uint16(img)

            with suppress_output():
                t0 = time.time()
                channels = [compress_channel_contextual(proc[...,i].astype(np.int32)) for i in range(3)]
                comp_time = time.time() - t0

            # write binary container
            bin_path = os.path.join(temp_dir, f"{base}_CAPRCppp.bin")
            with open(bin_path, "wb") as fb:
                fb.write(b"CAPX")
                for (d, ctx_symbols, ctx_counts, off, ch_h, ch_w) in channels:
                    fb.write(struct.pack("<IIq", ch_h, ch_w, off))
                    for ctx in range(4):
                        fb.write(struct.pack("<I", len(ctx_symbols[ctx])))
                        for s,c in zip(ctx_symbols[ctx], ctx_counts[ctx]):
                            fb.write(struct.pack("<qq", s, c))
                    fb.write(struct.pack("<I", len(d)))
                    fb.write(d)
            comp_kb = os.path.getsize(bin_path) / 1024.0

            # decompress from container
            t1 = time.time()
            rec_proc = np.zeros((h,w,3), dtype=np.uint16)
            with open(bin_path, "rb") as fb:
                fb.read(4)
                for ch in range(3):
                    ch_h, ch_w, off_r = struct.unpack("<IIq", fb.read(16))
                    ctx_symbols = []; ctx_counts = []
                    for _ in range(4):
                        n_sym = struct.unpack("<I", fb.read(4))[0]
                        syms=[]; cnts=[]
                        for _ in range(n_sym):
                            sv, cv = struct.unpack("<qq", fb.read(16))
                            syms.append(sv); cnts.append(cv)
                        ctx_symbols.append(syms); ctx_counts.append(cnts)
                    dlen = struct.unpack("<I", fb.read(4))[0]
                    data_bytes = fb.read(dlen)
                    rec_ch = decompress_channel_contextual(data_bytes, ctx_symbols, ctx_counts, off_r, ch_h, ch_w)
                    rec_proc[..., ch] = np.clip(rec_ch, 0, 65535).astype(np.uint16)
            dec_time = time.time() - t1

            rec_rgb = ycocg_to_rgb_uint8(rec_proc)
            rec_path = os.path.join(temp_dir, f"{base}_reconstructed.png")
            Image.fromarray(rec_rgb).save(rec_path)

            residual = np.abs(img.astype(np.int16) - rec_rgb.astype(np.int16)).mean(axis=2)
            mse = np.mean((img.astype(np.float32) - rec_rgb.astype(np.float32))**2)
            psnr = float('inf') if mse==0 else 20*np.log10(255.0/np.sqrt(mse))
            cr = orig_kb / comp_kb if comp_kb>0 else 0
            bpp = (comp_kb*1024*8) / (h*w*3) if (h*w)>0 else 0

            # JPEG-LS comparison removed
            jls_cr = jls_psnr = None

            # visuals
            fig, ax = plt.subplots(1,3, figsize=(14,5))
            ax[0].imshow(img); ax[0].set_title("Original")
            ax[1].imshow(rec_rgb); ax[1].set_title("Reconstructed")
            ax[2].imshow(residual, cmap="inferno"); ax[2].set_title("Residual Map")
            for a in ax: a.axis("off")
            fig.suptitle(f"{base}{ext} | CR={cr:.2f}× | PSNR={psnr:.2f} dB")
            visuals.append({
                "name": f"{base}_comparison.png",
                "bytes": fig_to_png_bytes(fig)
            })
            plt.close(fig)

            rows.append({
                "Image": f"{base}{ext}",
                "Resolution": f"{w}×{h}",
                "Original (KB)": f"{orig_kb:.2f}",
                "Compressed (KB)": f"{comp_kb:.2f}",
                "CR (CAP-RC++)": f"{cr:.2f}",
                "PSNR (CAP-RC++)": f"{psnr:.2f} dB",
                "CompTime": f"{comp_time:.2f}s",
                "DecompTime": f"{dec_time:.2f}s",
               
            })

            per_links.append({
                "Image": f"{base}{ext}",
                "Reconstructed PNG": rec_path,
                "Compressed BIN": bin_path
            })

        except Exception as e:
            st.warning(f"Error processing {getattr(f, 'name', str(f))}: {e}")

    if not rows:
        return {
            "df": pd.DataFrame(),
            "visuals": [],
            "csv_bytes": b"",
            "zip_bytes": b"",
            "per_items": [],
            "charts": []
        }

    # Dataframe for table
    df = pd.DataFrame(rows)
    csv_path = os.path.join(temp_dir, "CAPRCppp_Results.csv")
    df.to_csv(csv_path, index=False)

    # zip archive of temp dir
    zip_path = shutil.make_archive(os.path.join(temp_dir, "CAPRCppp_all"), "zip", temp_dir)

    # Charts (only if multiple images)
    charts = []
    if len(df) > 1:
        # numeric conversions for plotting
        def to_float_col(series):
            # remove non-numeric chars and coerce
            return pd.to_numeric(series.astype(str).str.replace(r"[^\d\.\-eE]", "", regex=True), errors="coerce").fillna(0.0)
        cr_col = to_float_col(df["CR (CAP-RC++)"])
        psnr_col = to_float_col(df["PSNR (CAP-RC++)"])
        # CR chart
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(df["Image"], cr_col)
        ax.set_ylabel("Compression Ratio (×)")
        ax.set_title("Compression Ratio (CAP-RC+++)")
        cp = os.path.join(temp_dir, "CR_chart.png"); fig.savefig(cp, bbox_inches="tight"); plt.close(fig)
        charts.append(cp)
        # PSNR chart
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(df["Image"], psnr_col)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("PSNR (CAP-RC++)")
        pp = os.path.join(temp_dir, "PSNR_chart.png"); fig.savefig(pp, bbox_inches="tight"); plt.close(fig)
        charts.append(pp)

        # JPEG-LS comparison charts removed

    with open(csv_path, "rb") as f_csv:
        csv_bytes = f_csv.read()
    with open(zip_path, "rb") as f_zip:
        zip_bytes = f_zip.read()

    per_items = []
    for p in per_links:
        rec_path = p["Reconstructed PNG"]
        bin_path = p["Compressed BIN"]
        with open(rec_path, "rb") as f_png:
            rec_png_bytes = f_png.read()
        with open(bin_path, "rb") as f_bin:
            bin_bytes = f_bin.read()
        per_items.append({
            "image": p["Image"],
            "rec_png_name": os.path.basename(rec_path),
            "rec_png_bytes": rec_png_bytes,
            "bin_name": os.path.basename(bin_path),
            "bin_bytes": bin_bytes
        })

    chart_items = []
    for chart_path in charts:
        with open(chart_path, "rb") as f_chart:
            chart_items.append({
                "name": os.path.basename(chart_path),
                "bytes": f_chart.read()
            })

    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "df": df,
        "visuals": visuals,
        "csv_bytes": csv_bytes,
        "zip_bytes": zip_bytes,
        "per_items": per_items,
        "charts": chart_items
    }


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="CAP-RC+++", layout="wide")
st.title("CAP-RC+++: Context-Adaptive Lossless Image Compression")
st.write("Upload one or more lossless images (BMP/TIFF). The app provides a summary table, visuals, and per-image downloads.")

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["bmp", "tif", "tiff", "png"],
    accept_multiple_files=True
)

if st.button("Start Compression", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one image.")
    else:
        with st.spinner("Compressing and reconstructing images..."):
            results = process_images(uploaded_files)

        df = results["df"]
        st.subheader("Summary Table")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV Summary",
            data=results["csv_bytes"],
            file_name="CAPRCppp_Results.csv",
            mime="text/csv"
        )
        st.download_button(
            "Download ZIP (All Files)",
            data=results["zip_bytes"],
            file_name="CAPRCppp_all.zip",
            mime="application/zip"
        )

        if results["visuals"]:
            st.subheader("Visual Comparison")
            for visual in results["visuals"]:
                st.download_button(
                    "Download comparison image",
                    data=visual["bytes"],
                    file_name=visual["name"],
                    mime="image/png",
                    key=f"visual_{visual['name']}"
                )

        st.subheader("Per-Image Downloads")
        for idx, item in enumerate(results["per_items"], start=1):
            with st.expander(item["image"], expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Reconstructed PNG",
                        data=item["rec_png_bytes"],
                        file_name=item["rec_png_name"],
                        mime="image/png",
                        key=f"png_{idx}_{item['rec_png_name']}"
                    )
                with col2:
                    st.download_button(
                        "Download Compressed BIN",
                        data=item["bin_bytes"],
                        file_name=item["bin_name"],
                        mime="application/octet-stream",
                        key=f"bin_{idx}_{item['bin_name']}"
                    )

        if results["charts"]:
            st.subheader("Charts")
            for chart in results["charts"]:
                st.download_button(
                    f"Download {chart['name']}",
                    data=chart["bytes"],
                    file_name=chart["name"],
                    mime="image/png",
                    key=f"chart_{chart['name']}"
                )
