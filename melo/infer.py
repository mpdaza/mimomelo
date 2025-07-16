import os
import click
from api import TTS
import loggers

    
    
@click.command()
@click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the checkpoint file")
@click.option('--text', '-t', type=str, default=None, help="Text to speak")
@click.option('--language', '-l', type=str, default="EN", help="Language of the model")
@click.option('--output_dir', '-o', type=str, default="outputs", help="Path to the output")
@click.option('--type', '-tp', 'type_name', type=click.Choice(['val', 'test', 'balanced'], case_sensitive=False), required=True, help="Inference split: val, test or balanced")
@click.option('--output_name', '-on', 'output_name', type=str, required=False, help="Custom output filename (no directory)")
def main(ckpt_path, text, language, output_dir, type_name, output_name):
    if ckpt_path is None:
        raise ValueError("The model_path must be specified")
    
    config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)
    
#     # Subdirectorio según el tipo
#     type_dir = os.path.join(output_dir, type_name.upper())
#     os.makedirs(type_dir, exist_ok=True)
    
#     for spk_name, spk_id in model.hps.data.spk2id.items():
#         save_path = f'{output_dir}/output.wav'
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         model.tts_to_file(text, spk_id, save_path)

    # Carpeta de salida: outputs/VAL o outputs/TEST...
    type_name = type_name.upper()
    type_dir = os.path.join(output_dir, type_name)
    os.makedirs(type_dir, exist_ok=True)
    
    model_name = os.path.splitext(os.path.basename(ckpt_path))[0]  # e.g. G_1000
    for i, (spk_name, spk_id) in enumerate(model.hps.data.spk2id.items(), start=1):
        # Nombre del archivo: si se proporciona, se usa ese
        if output_name:
            save_path = os.path.join(type_dir, output_name)
        else:
            filename = f"{type_name.lower()}_{model_name}_{i}.wav"
            save_path = os.path.join(type_dir, filename)

        print(f"[INFO] Guardando archivo: {save_path}")
        model.tts_to_file(text, spk_id, save_path)

if __name__ == "__main__":
    loggers.init()
    # logger.log_general.info("Glue the sheet to the dark blue background.")
    # logger.log_general.info("G_229000.pth")
    main()
