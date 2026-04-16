"""
ZipEnhancer Module - Audio Denoising Enhancer

Drop-in replacement for the original ModelScope-based ZipEnhancer, now backed by
`DeepFilterNet <https://github.com/Rikorose/DeepFilterNet>`_ — a fully open-source,
PyPI-installable speech enhancement library that requires no ModelScope account and
downloads its pre-trained weights automatically from the internet.

Install the backend with::

    pip install deepfilternet

The ``model_path`` parameter is kept for API compatibility but is repurposed:
it now accepts a DeepFilterNet model variant name (``"DeepFilterNet"``,
``"DeepFilterNet2"``, or ``"DeepFilterNet3"``). The default (``None``) loads
DeepFilterNet3, the best-quality variant.
"""

import os
import tempfile
from typing import Optional

import numpy as np
import torchaudio


class ZipEnhancer:
    """Audio denoising enhancer powered by DeepFilterNet.

    This is a drop-in replacement for the original ModelScope-based ZipEnhancer.
    It preserves the same ``enhance(input_path, output_path, normalize_loudness)``
    interface while removing the hard dependency on ``modelscope``.

    Requires::

        pip install deepfilternet
    """

    # DeepFilterNet native sample rate
    _DF_SR = 48_000

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the DeepFilterNet enhancer.

        Args:
            model_path: DeepFilterNet model variant to load.  Accepted values:
                ``"DeepFilterNet"``, ``"DeepFilterNet2"``, ``"DeepFilterNet3"``,
                or a local path to a model directory produced by DeepFilterNet.
                Defaults to ``None``, which loads the best default model
                (currently ``"DeepFilterNet3"``).

                .. note::
                    For API compatibility with the original ModelScope-based
                    ZipEnhancer, the old ModelScope model ID
                    ``"iic/speech_zipenhancer_ans_multiloss_16k_base"`` is
                    silently remapped to the DeepFilterNet3 default.
        """
        try:
            from df.enhance import enhance, init_df
        except ImportError as exc:
            raise ImportError(
                "ZipEnhancer now requires the 'deepfilternet' package. "
                "Install it with: pip install deepfilternet"
            ) from exc

        # Remap legacy ModelScope model IDs to the DeepFilterNet default
        _LEGACY_IDS = {
            "iic/speech_zipenhancer_ans_multiloss_16k_base",
            "iic/speech_zipenhancer_ans_multiloss_16k_base_onnx",
        }
        if model_path in _LEGACY_IDS or model_path is None:
            model_base_dir = None  # let DeepFilterNet choose the best default
        else:
            model_base_dir = model_path

        self._enhance_fn = enhance
        self._model, self._df_state, _ = init_df(model_base_dir=model_base_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_loudness(self, wav_path: str) -> None:
        """In-place LUFS loudness normalisation to −20 LUFS."""
        audio, sr = torchaudio.load(wav_path)
        loudness = torchaudio.functional.loudness(audio, sr)
        normalized_audio = torchaudio.functional.gain(audio, -20.0 - loudness)
        torchaudio.save(wav_path, normalized_audio, sr)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enhance(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        normalize_loudness: bool = True,
    ) -> str:
        """Denoise an audio file and return the path to the enhanced output.

        Args:
            input_path: Path to the noisy input audio file (any format
                supported by ``torchaudio``).
            output_path: Destination path for the enhanced WAV file.
                A temporary file is created when ``None``.
            normalize_loudness: If ``True``, apply LUFS loudness normalisation
                (target −20 LUFS) to the output.

        Returns:
            str: Path to the enhanced output file.

        Raises:
            FileNotFoundError: If ``input_path`` does not exist.
            RuntimeError: If enhancement fails.
        """
        from df.enhance import load_audio, save_audio

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file does not exist: {input_path}")

        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                output_path = tmp.name

        try:
            # DeepFilterNet expects audio at its native sample rate (48 kHz).
            audio, _ = load_audio(input_path, sr=self._df_state.sr())

            enhanced = self._enhance_fn(self._model, self._df_state, audio)

            save_audio(output_path, enhanced, self._df_state.sr())

            if normalize_loudness:
                self._normalize_loudness(output_path)

            return output_path

        except Exception as exc:
            # Clean up any partially-written temporary file
            if output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"Audio denoising processing failed: {exc}") from exc
