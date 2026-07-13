"""Font and subtitle style utility functions for Gradio UI."""
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import gradio as gr

class StylePreview:
    def __init__(self, fonts_map, fonts_default):
        self.fonts_map = fonts_map
        self.fonts_default = fonts_default

    def update_style_controls(self, style):
        """Update the visibility of style controls based on the selected style."""
        visible_update = gr.update(visible=style != "None")
        return visible_update, visible_update

    def update_subtitle_preview(self, font_name, font_size, primary_hex, outline_w,
                              outline_hex, subtitle_style):
        """Generate a preview image of the subtitle style.

        Uses a portrait aspect ratio (9:16) to match typical short-form video,
        with the subtitle positioned at the bottom like the actual output.
        Font size is rendered at the user-specified pixel size so the preview
        closely represents the final result.
        """
        try:
            # Portrait 9:16 preview to match short-form video format
            preview_w, preview_h = 200, 354
            img = Image.new('RGB', (preview_w, preview_h), color='#2a2a2a')
            draw = ImageDraw.Draw(img)

            # Load font — scale to match the /250 factor used in run_pipeline
            font_path = self.fonts_map.get(font_name, self.fonts_map[self.fonts_default])
            preview_font_size = max(1, int(int(font_size) * 200 / 250))
            try:
                font = ImageFont.truetype(font_path, preview_font_size)
            except Exception:
                font = ImageFont.load_default()

            sample_text = "Sample Subtitle"
            bbox = draw.textbbox((0, 0), sample_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position at bottom-center, like actual subtitles
            x = (preview_w - text_width) // 2
            y = preview_h - text_height - 30

            # Draw background/outline based on style
            if subtitle_style == "Box":
                padding = int(outline_w) + 4
                draw.rectangle(
                    [x - padding, y - padding,
                     x + text_width + padding, y + text_height + padding],
                    fill=outline_hex
                )
            elif subtitle_style == "Outline" and outline_w:
                for ox in range(-int(outline_w), int(outline_w) + 1):
                    for oy in range(-int(outline_w), int(outline_w) + 1):
                        if ox * ox + oy * oy <= int(outline_w) ** 2:
                            draw.text((x + ox, y + oy), sample_text,
                                      font=font, fill=outline_hex)

            # Draw main text
            draw.text((x, y), sample_text, font=font, fill=primary_hex)

            return img
        except Exception as e:
            print(f"Preview generation error: {e}")
            return None
