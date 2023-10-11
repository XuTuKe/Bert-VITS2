import asyncio
from enum import Enum
import os
import sys
from fastapi import FastAPI, Response
from io import BytesIO
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel
import torch
from av import open as avopen

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
from scipy.io import wavfile
import logging

app = FastAPI()

def log_device_usage(msg, use_cuda=True):
    import psutil
    mem_Mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    cuda_mem_Mb = torch.cuda.memory_allocated(0) / 1024 ** 2 if use_cuda else 0
    logging.info(f"{msg}:, mem: {int(mem_Mb)}Mb, gpu mem:{int(cuda_mem_Mb)}Mb")


def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    phone = commons.intersperse(phone, 0)
    tone = commons.intersperse(tone, 0)
    language = commons.intersperse(language, 0)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1

    bert = get_bert(norm_text, word2ph, language_str, device=dev)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JA":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language):
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    with torch.no_grad():
        x_tst = phones.to(dev).unsqueeze(0)
        tones = tones.to(dev).unsqueeze(0)
        lang_ids = lang_ids.to(dev).unsqueeze(0)
        bert = bert.to(dev).unsqueeze(0)
        ja_bert = ja_bert.to(dev).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(dev)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(dev)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
        torch.cuda.empty_cache()        
        return audio


def replace_punctuation(text, i=2):
    punctuation = "，。？！"
    for char in punctuation:
        text = text.replace(char, char * i)
    return text


def wav2(i, o, format):
    inp = avopen(i, "rb")
    out = avopen(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


# Load Generator
hps = utils.get_hparams_from_file("./configs/config.json")

if sys.platform == "darwin" and torch.backends.mps.is_available():
    dev = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    dev = "cuda"
    
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model,
).to(dev)
_ = net_g.eval()

_ = utils.load_checkpoint("logs/genshin_mix/G_19000.pth", net_g, None, skip_optimizer=True)


def tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language):
    slices = text.split("|")
    audio_list = []
    log_device_usage("=============== before text infer")
    with torch.no_grad():
        for slice in slices:
            audio = infer(slice, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale, sid=speaker, language=language)
            audio_list.append(audio)
            silence = np.zeros(hps.data.sampling_rate)  # 生成1秒的静音
            audio_list.append(silence)  # 将静音添加到列表中
    audio_concat = np.concatenate(audio_list)
    log_device_usage("=============== after text infer")
    return (hps.data.sampling_rate, audio_concat)


class Language(str, Enum):
    zh = "ZH"
    jp = "JP"    

class AudioFormat(str,Enum):
    wav = "wav"
    mp3 = "mp3"
    
loop = asyncio.get_event_loop()

# 使用 json 传参数时
class TTSParams(BaseModel):
    text:str
    fmt:AudioFormat=AudioFormat.wav
    speaker:str="bcsz"
    sdp_ratio:float=0.2
    noise_scale:float=0.5
    noise_scale_w:float=0.6
    length_scale:float=1.2
    language:Language=Language.zh    
    
@app.post("/tts")
async def request_tts(text:str, fmt:AudioFormat=AudioFormat.wav, speaker:str="bcsz", sdp_ratio:float=0.2, noise_scale:float=0.5, noise_scale_w:float=0.6, length_scale:float=1.2, language:Language=Language.zh):
    text = text.replace("/n", "")
    rate, audio = await loop.run_in_executor(None, tts_fn,
        text, 
        speaker,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        language
    )

    with BytesIO() as wav:
        wavfile.write(wav, rate, audio)
        torch.cuda.empty_cache()
        if fmt == "wav":
            response =  Response(wav.getvalue(), media_type="audio/wav")
            response.headers["Content-Disposition"] = 'attachment; filename="audio.wav"'
            return response
        wav.seek(0, 0)
        with BytesIO() as ofp:
            wav2(wav, ofp, fmt)
            response = Response(
                ofp.getvalue(), media_type="audio/mpeg" if fmt == "mp3" else "audio/ogg"
            )
            response.headers["Content-Disposition"] = 'attachment; filename="audio.mp3"'
            return response
               

if __name__ == '__main__':
    
    import uvicorn
    # 等于通过 uvicorn 命令行 uvicorn 脚本名:app对象 启动服务：
    #  uvicorn xxx:app --reload
    uvicorn.run('fast_server:app',host="127.0.0.1", port=8001, reload=True)
