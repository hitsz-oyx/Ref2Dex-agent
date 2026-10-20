"""Label Gym camera frames and encode a bounded, four-times-slow MANO preview."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    output = args.preview.resolve()
    if (output / 'preview-slow4x.mp4').exists():
        raise ValueError('encoded video already exists; use a fresh preview output')
    metadata = json.loads((output / 'manifest.json').read_text())
    render = json.loads((output / 'render-manifest.json').read_text())
    if render['status'] != 'COMPLETED' or render['fk_max_mm'] > 2:
        raise ValueError('native FK verification must pass before publishing')
    with np.load(output / 'camera-frames.npz') as archive:
        frames = archive['frames'].copy()
    if frames.shape[0] != 25:
        raise ValueError('full current plus24 forecast frames required')
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 16)
    width, height = frames.shape[2], frames.shape[1]
    images = []
    for index, frame in enumerate(frames):
        image = Image.new('RGB', (width, height + 150), '#151b24')
        image.paste(Image.fromarray(frame), (0, 70))
        draw = ImageDraw.Draw(image)
        draw.text((20, 10), 'Measured teacher / native Inspire', font=font, fill='#dde4ec')
        draw.text((width // 2 + 20, 10), 'WAM forecast / fitted MANO surface', font=font, fill='#40d4eb')
        draw.text((20, 40), 'Same measured object replay and camera; collision-free visual comparison', font=small, fill='#a9b7c8')
        error = metadata['per_frame_fit_mean_mm'][index]
        raw = metadata['per_frame_prediction_error_mm'][index]
        draw.text((20, height + 80), 'Fixed forecast at tick %d   |   +%.3f s   |   playback 4x slower' % (metadata['tick'], index / 30), font=small, fill='white')
        draw.text((20, height + 105), 'MANO fit residual %.1f mm   |   WAM vs future teacher %.1f mm   |   gold: object path / local goal' % (error, raw), font=small, fill='#f7cc6b')
        draw.text((20, height + 128), 'MANO mesh is an approximate fit to11 points; future teacher states are comparison only.', font=small, fill='#a9b7c8')
        images.append(image)
    images[0].save(output / 'preview-current.png')
    images[12].save(output / 'preview-middle.png')
    images[24].save(output / 'preview-end.png')
    contact = Image.new('RGB', (width, images[0].height * 3))
    for row, index in enumerate([0, 12, 24]):
        contact.paste(images[index], (0, images[index].height * row))
    contact.resize((960, round(contact.height * 960 / width))).save(output / 'preview-contact-sheet.png')
    command = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % images[0].size,
               '-r', '30', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output / 'preview-slow4x.mp4')]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    timeline = [images[0]] * 15 + [image for image in images for _ in range(4)] + [images[-1]] * 30
    for image in timeline:
        encoder.stdin.write(np.asarray(image).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')
    size = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    if size > 150 * 1024 ** 2:
        raise ValueError('preview artifact budget exceeded')
    print(json.dumps(dict(status='COMPLETED', output=str(output), total_bytes=size,
                          video='preview-slow4x.mp4', still='preview-middle.png')), flush=True)


if __name__ == '__main__':
    main()
