import re
import json
import torch
import soundfile
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch

from .api import TTS

class MultilingualTTS(nn.Module):
    def __init__(self, 
                narr_language, narr_speaker, narr_speed, 
                device='auto',
                ml_tts_config_path=None):
        super().__init__()
        assert ml_tts_config_path
        self._read_config(ml_tts_config_path)

        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()
        self.device = device
        self.narr_language = narr_language
        self.narr_speaker = narr_speaker
        self.narr_speed = narr_speed
        self._load_tts_models()


    def _read_config(self, ml_config_path):
        with open(ml_config_path, "r", encoding="utf-8") as f:
            data = f.read()
            self.ml_config = json.loads(data)


    def _load_tts_models(self):
        cfg_chk_map = {}
        self.tts_models = {}  # Cache for loaded tts models
        for lang_code, lang_variants in self.ml_config.items():
            if lang_code == 'None':
                lang_code = self.narr_language
            for lang_variant, model_paths in lang_variants.items():
                if lang_code == self.narr_language and lang_variant == 'None':
                    lang_variant = self.narr_speaker
                model_config = model_paths["config"]
                model_checkpoint = model_paths["checkpoint"]
                cache_key = (model_config, model_checkpoint)
                if cache_key in cfg_chk_map:
                    tts = cfg_chk_map[cache_key]
                else:
                    tts = TTS(language=lang_code, device=self.device, use_hf=False, ckpt_path=model_checkpoint, config_path=model_config)
                    cfg_chk_map[cache_key] = tts
                self.tts_models[(lang_code, lang_variant)] = tts


    def _split_into_participant_segments(self, input_text, quiet=False):
        language_regex = r'\[([A-Za-z-]+)(?:,\s*([A-Za-z-]+))?\](.*?)\[\1(?:-[A-Za-z-]+)?(?:,\s*[A-Za-z-]+)?\]'
        segments = []
        matches = re.finditer(language_regex, input_text)
        prev_end = 0
        for match in matches:
            start_idx = match.start()
            end_idx = match.end()
            if start_idx > prev_end and len(input_text[prev_end:start_idx].strip()) > 0:
                segments.append([input_text[prev_end:start_idx], [self.narr_language, self.narr_speaker]])
            if len(match.group(3).strip()) > 0:
                segments.append([match.group(3), [match.group(1), match.group(2)]])
            prev_end = end_idx
        if prev_end < len(input_text) and len(input_text[prev_end:].strip()) > 0:
            segments.append([input_text[prev_end:], [self.narr_language, self.narr_speaker]])
        if not quiet:
            print(" > Text participants splits:")
            print(segments)
            print(" > ===========================")
        return segments


    def tts_to_file(self, text, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, pbar=None, format=None, position=None, quiet=False,):
        participant_segments = self._split_into_participant_segments(text, quiet)
        audio_list = []
        if pbar:
            tx = pbar(participant_segments)
        else:
            if position:
                tx = tqdm(participant_segments, position=position)
            elif quiet:
                tx = participant_segments
            else:
                tx = tqdm(participant_segments)
        tts_model = None
        for t in tx:
            lang_code = t[1][0]
            speaker_id = t[1][1] 
            speed = self.narr_speed if lang_code == self.narr_language and speaker_id == self.narr_speaker else 1
            tts_model = self.tts_models[(lang_code, speaker_id)]
            speaker_ids = tts_model.hps.data.spk2id
            spkr = speaker_ids[speaker_id]
            part_text = t[0]
            audio = tts_model.tts_to_file(part_text, spkr, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w, speed=speed, pbar=pbar, position=position, quiet=quiet)
            audio_list.append(audio)
        audio = TTS.audio_numpy_concat(audio_list, sr=tts_model.hps.data.sampling_rate)
        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, tts_model.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, tts_model.hps.data.sampling_rate)
