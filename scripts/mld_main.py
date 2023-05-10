import gradio as gr

from mldanbooru.interface import Infer
from webui import wrap_gradio_gpu_call
from modules import script_callbacks
from modules.ui import setup_progressbar

def build_ui(infer):
    with gr.Blocks() as mld:
        dummy_component = gr.Label(visible=False)
        with gr.Tabs():
            with gr.Tab('Tag one image'):
                with gr.Row():
                    with gr.Column():
                        gr_input_image = gr.Image(type='pil', label='Original Image')
                        with gr.Row():
                            gr_threshold = gr.Slider(0.0, 1.0, 0.7, label='Tagging Confidence Threshold')
                            gr_image_size = gr.Slider(128, 960, 448, step=32, label='Image for Recognition')
                            gr_keep_ratio = gr.Checkbox(value=False, label='Keep the Ratio')
                        with gr.Row():
                            gr_model = gr.Dropdown(infer.MODELS, value=infer.DEFAULT_MODEL, label='Model')
                        with gr.Row():
                            gr_space = gr.Checkbox(value=True, label='Use Space Instead Of _')
                            gr_escape = gr.Checkbox(value=True, label='Use Text Escape')
                            gr_conf = gr.Checkbox(value=False, label='With confidence')

                        with gr.Row():
                            gr_btn_submit = gr.Button(value='Tagging', variant='primary')
                            gr_btn_unload = gr.Button(value='Unload')

                    with gr.Column():
                        with gr.Tabs():
                            with gr.Tab("Tags"):
                                gr_tags = gr.Label(label='Tags')
                            with gr.Tab("Exported Text"):
                                gr_output_text = gr.TextArea(label='Exported Text', lines=10)
                            gr_info = gr.Text(value="", show_label=False)

            with gr.Tab('Tag images'):
                with gr.Row():
                    with gr.Column():
                        gr_m_input_dir = gr.Textbox(label='Images path')
                        with gr.Row():
                            gr_m_threshold = gr.Slider(0.0, 1.0, 0.7, label='Tagging Confidence Threshold')
                            gr_m_image_size = gr.Slider(128, 960, 448, step=32, label='Image for Recognition')
                            gr_m_keep_ratio = gr.Checkbox(value=False, label='Keep the Ratio')
                        with gr.Row():
                            gr_m_model = gr.Dropdown(infer.MODELS, value=infer.DEFAULT_MODEL, label='Model')
                        with gr.Row():
                            gr_m_space = gr.Checkbox(value=True, label='Use Space Instead Of _')
                            gr_m_escape = gr.Checkbox(value=True, label='Use Text Escape')
                            gr_m_out_type = gr.Dropdown(['txt', 'json'], value='txt', label='File type')

                        with gr.Row():
                            gr_m_btn_submit = gr.Button(value='Tagging', variant='primary')
                            gr_m_btn_unload = gr.Button(value='Unload')

                    with gr.Column():
                        mld_progress = gr.HTML(elem_id="mld_progress", value="")
                        gr_m_info = gr.Text(value="", show_label=False)

        gr_btn_submit.click(
            infer.infer_one,
            inputs=[
                gr_input_image, gr_threshold, gr_image_size,
                gr_keep_ratio, gr_model,
                gr_space, gr_escape, gr_conf
            ],
            outputs=[gr_output_text, gr_tags],
        )
        gr_btn_unload.click(
            infer.unload,
            inputs=[],
            outputs=[gr_info],
        )
        gr_m_btn_submit.click(
            wrap_gradio_gpu_call(infer.infer_folder, extra_outputs=[gr.update()]),
            _js="mld_tagging",
            inputs=[
                dummy_component,
                gr_m_input_dir, gr_m_threshold, gr_m_image_size,
                gr_m_keep_ratio, gr_m_model,
                gr_m_space, gr_m_escape, gr_m_out_type
            ],
            outputs=[gr_m_info],
        )
        gr_m_btn_unload.click(
            infer.unload,
            inputs=[],
            outputs=[gr_m_info],
        )
    return mld

def on_ui_tabs():
    infer = Infer()
    mld = build_ui(infer)
    return [(mld, "MLDanbooru Tagger", "mld_tagger")]

script_callbacks.on_ui_tabs(on_ui_tabs)