"""
app.py — NeuralRestore Gradio UI
================================
Beautiful Before/After image restoration interface.
Runs locally — no internet needed after setup.
"""

import os
import gc
import time
from pathlib import Path
from PIL import Image
import numpy as np

import gradio as gr

from src.pipeline import NeuralRestorePipeline
from src.config import WEIGHTS_DIR, LABEL_NAMES, LABEL_ICONS

# ────────────────────────────────────────────────────────────────────
# Load Pipeline (once at startup)
# ────────────────────────────────────────────────────────────────────
print("Loading NeuralRestore Pipeline...")
pipeline = NeuralRestorePipeline(weights_dir=WEIGHTS_DIR)
print("Pipeline ready!\n")


# ────────────────────────────────────────────────────────────────────
# Core Restore Function
# ────────────────────────────────────────────────────────────────────
def restore_image(image, apply_sr, mask_threshold, mask_dilation, use_tta):
    """
    Main function called by Gradio when user clicks Restore.
    """
    if image is None:
        return None, "❌ Please upload an image first!", ""

    start_time = time.time()

    try:
        # Run pipeline
        result = pipeline.restore(
            image,
            apply_sr=apply_sr,
            mask_threshold=mask_threshold,
            mask_dilation=int(mask_dilation),
            use_tta=use_tta,
        )

        restored_np = result["restored"]
        labels      = result["labels"]
        damage_found = result["damage_found"]
        steps       = result["steps_applied"]

        # Build damage report
        report = build_report(labels, damage_found, steps,
                              result["original"], restored_np)

        elapsed = time.time() - start_time
        time_text = f"⏱️ Processing time: {elapsed:.1f} seconds"

        return Image.fromarray(restored_np), report, time_text

    except Exception as e:
        error_msg = f"❌ Error during restoration:\n{str(e)}"
        return None, error_msg, ""


def build_report(labels, damage_found, steps, original_np, restored_np):
    """Build a formatted damage analysis report."""

    lines = ["## 📊 Damage Analysis Report\n"]

    if damage_found:
        detected = [k for k in LABEL_NAMES if labels.get(k, 0) == 1]
        lines.append(f"**Status:** ⚠️ Damage detected\n")
        lines.append(f"**Types found:** {', '.join(detected)}\n")
    else:
        lines.append(f"**Status:** ✅ No damage detected\n")

    lines.append("\n---\n")

    lines.append("### 🔍 Damage Type Analysis\n")
    for name in LABEL_NAMES:
        conf  = labels.get(f"{name}_confidence", 0.0)
        found = labels.get(name, 0) == 1
        icon  = LABEL_ICONS.get(name, "•")
        bar   = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        status = "✅ Detected" if found else "⬜ Not found"
        lines.append(
            f"{icon} **{name.capitalize()}**: {status}  \n"
            f"   Confidence: `{bar}` {conf*100:.1f}%\n\n"
        )

    lines.append("---\n")

    lines.append("### ⚙️ Processing Steps Applied\n")
    step_icons = {
        "damage_detection" : "🔍 Damage Detection",
        "inpainting"       : "🖌️ Damage Removal (Inpainting)",
        "denoising"        : "✨ Noise Removal",
        "deblurring"       : "🔭 Blur Removal (Sharpening)",
        "super_resolution" : "📐 Super Resolution (4x HD)",
    }
    for step in steps:
        lines.append(f"- {step_icons.get(step, step)}\n")

    lines.append("\n---\n")

    h_orig, w_orig = original_np.shape[:2]
    h_rest, w_rest = restored_np.shape[:2]
    lines.append("### 📐 Image Info\n")
    lines.append(f"- **Input size:** {w_orig} × {h_orig} px\n")
    lines.append(f"- **Output size:** {w_rest} × {h_rest} px\n")
    if h_rest > h_orig:
        scale = h_rest / h_orig
        lines.append(f"- **Upscale factor:** {scale:.1f}x\n")

    return "".join(lines)


# ────────────────────────────────────────────────────────────────────
# Gradio UI
# ────────────────────────────────────────────────────────────────────
def create_ui():
    with gr.Blocks(
        title = "NeuralRestore — AI Image Restoration",
        theme = gr.themes.Soft(
            primary_hue   = "blue",
            secondary_hue = "slate",
        ),
        css = """
            .title-text {
                text-align: center;
                font-size: 2.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            .subtitle-text {
                text-align: center;
                color: #64748b;
                font-size: 1.1rem;
                margin-bottom: 2rem;
            }
            .restore-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                border: none !important;
                color: white !important;
                font-size: 1.1rem !important;
                font-weight: 600 !important;
                padding: 0.75rem 2rem !important;
                border-radius: 0.75rem !important;
            }
            .restore-btn:hover {
                opacity: 0.9 !important;
                transform: translateY(-1px) !important;
            }
            .tip-box {
                background: #f0f9ff;
                border-left: 4px solid #667eea;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-top: 1rem;
            }
        """
    ) as demo:

        # ── Header ───────────────────────────────────────────────────
        gr.HTML("""
            <div class="title-text">🖼️ NeuralRestore</div>
            <div class="subtitle-text">
                AI-Powered Image Restoration Pipeline<br>
                Detects and removes scratches, stains, cracks, dust, noise and blur automatically
            </div>
        """)

        # ── Main Row ─────────────────────────────────────────────────
        with gr.Row():

            # Left Column — Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Damaged Image")

                input_image = gr.Image(
                    label   = "Damaged Image",
                    type    = "pil",
                    height  = 400,
                    sources = ["upload", "clipboard"],
                )

                apply_sr = gr.Checkbox(
                    label   = "Apply 4x Super Resolution (HD upscaling)",
                    value   = True,
                    info    = "Upscales output to 4x the input size using Real-ESRGAN"
                )

                gr.Markdown("#### 🎯 Mask Detection Settings")
                mask_threshold = gr.Slider(
                    minimum = 0.05, maximum = 0.90, value = 0.25, step = 0.05,
                    label   = "Detection Threshold",
                    info    = "Lower = detect more damage. Try 0.10–0.15 for subtle stains/dust."
                )
                mask_dilation = gr.Slider(
                    minimum = 0, maximum = 15, value = 5, step = 1,
                    label   = "Mask Dilation (px)",
                    info    = "Expands the detected mask. Helps LaMa blend edges cleanly."
                )
                use_tta = gr.Checkbox(
                    label = "Use TTA (better masks, slightly slower)",
                    value = True,
                    info  = "Runs 4 flip variants to catch damage in any orientation."
                )

                restore_btn = gr.Button(
                    "🔄 Restore Image",
                    variant     = "primary",
                    elem_classes = ["restore-btn"],
                    size        = "lg",
                )

                time_text = gr.Markdown("")

                gr.HTML("""
                    <div class="tip-box">
                        <b>💡 Tips:</b><br>
                        • Works on photos, scanned documents, old pictures<br>
                        • Color, grayscale, or sepia images all supported<br>
                        • Larger images may take longer to process<br>
                        • Uncheck SR for faster processing
                    </div>
                """)

            # Right Column — Output
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Restored Image")

                output_image = gr.Image(
                    label  = "Restored Image",
                    type   = "pil",
                    height = 400,
                )

                download_btn = gr.DownloadButton(
                    label = "⬇️ Download Restored Image",
                    value = None,
                )

        # ── Report Row ────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Damage Analysis Report")
                report_text = gr.Markdown(
                    value = "*Upload an image and click Restore to see the analysis*",
                )

        # ── Examples ─────────────────────────────────────────────────
        examples_list = []
        samples_dir = Path("data/samples")
        if samples_dir.exists():
            sample_files = sorted(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")))
            for f in sample_files:
                examples_list.append([str(f), True])
        
        # Fallback to test dataset if samples not found
        if not examples_list:
            fallback_dir = Path("data/dataset/model1/test/damaged")
            if fallback_dir.exists():
                dataset_files = sorted(list(fallback_dir.glob("*.png")) + list(fallback_dir.glob("*.jpg")))
                # Pick a representative subset of up to 5 files
                for f in dataset_files:
                    if any(x in f.name for x in ["scratch", "stain", "crack", "combined", "both"]):
                        examples_list.append([str(f), True])
                    if len(examples_list) >= 5:
                        break
                # Fallback to any first 5 files if still empty
                if not examples_list:
                    for f in dataset_files[:5]:
                        examples_list.append([str(f), True])

        if examples_list:
            gr.Markdown("---")
            gr.Markdown("### 🖼️ Try with Sample Images")
            gr.Examples(
                examples = examples_list,
                inputs  = [input_image, apply_sr],
                label   = "Select a sample damaged image to test the restoration pipeline:",
                cache_examples = False,
            )

        # ── Footer ────────────────────────────────────────────────────
        gr.HTML("""
            <div style="text-align:center; color:#94a3b8; margin-top:2rem; font-size:0.9rem;">
                NeuralRestore | PyTorch | MIT License<br>
                Models: EfficientNet-B4 · LaMa · DnCNN · NAFNet · Real-ESRGAN
            </div>
        """)

        # ── Event Handlers ────────────────────────────────────────────
        def on_restore(image, apply_sr, mask_threshold, mask_dilation, use_tta):
            restored_pil, report, time_text = restore_image(
                image, apply_sr, mask_threshold, mask_dilation, use_tta
            )

            download_path = None
            if restored_pil is not None:
                download_path = "outputs/restored_output.png"
                os.makedirs(os.path.dirname(download_path), exist_ok=True)
                restored_pil.save(download_path)

            return restored_pil, report, time_text, download_path

        restore_btn.click(
            fn      = on_restore,
            inputs  = [input_image, apply_sr, mask_threshold, mask_dilation, use_tta],
            outputs = [output_image, report_text, time_text, download_btn],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        show_error  = True,
        inbrowser   = True,
    )
