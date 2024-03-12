import click
import os

@click.command
@click.argument('text')
@click.argument('output_path')
@click.option("--file", '-f', is_flag=True, show_default=True, default=False, help="Input text file to synthesize")
@click.option('--ml_tts_config', '-ml', type=str, default=None, help="Path to the ml-tts-config.json file with all language/speaker specifiers required for synthesis")
@click.option('--narr_language', '-l', default='EN', help='Narrator Language, defaults to English', type=click.Choice(['EN', 'ES', 'FR', 'ZH', 'JP', 'KR'], case_sensitive=False))
@click.option('--narr_speaker', '-spk', default='EN-Default', help='Narrator Speaker ID, only for English, leave empty for default, ignored if not English. If English, defaults to "EN-Default"', type=click.Choice(['EN-Default', 'EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU']))
@click.option('--narr_speed', '-s', default=1.0, help='Narrator Speed, defaults to 1.0', type=float)
@click.option('--device', '-d', default='auto', help='Device, defaults to auto')
def main(text, output_path, file, ml_tts_config, narr_language, narr_speaker, narr_speed, device):
    assert text
    if not os.path.exists(text):
        raise FileNotFoundError(f'--file/-f {file}: input text file not found.')
    else:
        with open(text) as f:
            text = f.read().strip()
    if text == '':
        raise ValueError('The input text file was empty.')
    assert ml_tts_config
    if not os.path.exists(ml_tts_config):
        raise FileNotFoundError(f'--ml_config/-ml {ml_tts_config}: multilingual config file not found.')
    from melo.ml_api import MultilingualTTS
    model = MultilingualTTS(narr_language, narr_speaker, narr_speed, device=device, ml_tts_config_path=ml_tts_config)
    model.tts_to_file(text, output_path)

if __name__ == "__main__":
    main()
