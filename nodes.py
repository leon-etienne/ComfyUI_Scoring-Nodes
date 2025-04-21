import torch
import clip
from torchvision import transforms
import lpips
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO

# aesthetic predictor import
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

# Device & models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
lpips_model = lpips.LPIPS(net='alex').to(device)

# Aesthetic predictor
aest_model, aest_preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
aest_model = aest_model.to(torch.bfloat16).to(device)
aest_model.eval()

# ——— Utility functions ———

def _to_pil(img):
    arr = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[2] in (1,3):
        hwc = arr
    elif arr.ndim == 3 and arr.shape[0] in (1,3):
        hwc = np.transpose(arr, (1,2,0))
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if hwc.ndim == 2:
        pass
    elif hwc.shape[2] == 1:
        hwc = hwc[:,:,0]
    hwc = np.clip(hwc * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(hwc)

def _pil_to_nhwc_tensor(pil):
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)

def _encode_texts(texts):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        return clip_model.encode_text(tokens)

def _encode_image(pil):
    img = clip_preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return clip_model.encode_image(img)

def _encode_images(pils):
    batch = torch.cat([clip_preprocess(im).unsqueeze(0) for im in pils], dim=0).to(device)
    with torch.no_grad():
        return clip_model.encode_image(batch)

# ——— CLIP‐based nodes ———

class TextSimilarity:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text1": ("STRING",), "text2": ("STRING",)}}
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("similarity",)
    FUNCTION = "calc"
    CATEGORY = "text/similarity"

    def calc(self, text1, text2):
        feats = _encode_texts([text1, text2])
        sim = torch.cosine_similarity(feats[0:1], feats[1:2]).item()
        return (sim,)

class ImageSimilarity:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image1": ("IMAGE",), "image2": ("IMAGE",)}}
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("similarity",)
    FUNCTION = "calc"
    CATEGORY = "image/similarity"

    def calc(self, image1, image2):
        pil1, pil2 = _to_pil(image1), _to_pil(image2)
        f1, f2 = _encode_image(pil1), _encode_image(pil2)
        sim = torch.cosine_similarity(f1, f2).item()
        return (sim,)

class MultiTextToImageSimilarity:
    @classmethod
    def INPUT_TYPES(cls):
        req = {"image": ("IMAGE",)}
        opt = {f"text{i}": ("STRING",) for i in range(1,9)}
        return {"required": req, "optional": opt}
    RETURN_TYPES = ("FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("highscore", "best_text", "plot")
    FUNCTION = "calc"
    CATEGORY = "multi/similarity"

    def calc(self, image, **texts):
        txts = [texts[f"text{i}"] for i in range(1,9) if texts.get(f"text{i}")]
        if not txts:
            return (0.0, "", None)
        pil = _to_pil(image)
        t_feats = _encode_texts(txts)
        i_feat = _encode_image(pil).repeat(len(txts),1)
        sims = torch.cosine_similarity(t_feats, i_feat, dim=1).cpu().tolist()
        idx = int(torch.tensor(sims).argmax())
        fig, ax = plt.subplots(figsize=(len(sims)*1.2,4), dpi=150)
        ax.bar(range(len(sims)), sims)
        ax.set_xticks(range(len(sims)))
        ax.set_xticklabels(txts, rotation=45, ha='right')
        ax.set_ylabel("Cosine similarity")
        ax.set_title("Text vs Image similarities")
        plt.tight_layout()
        buf = BytesIO(); fig.savefig(buf, format='PNG', dpi=150); plt.close(fig)
        buf.seek(0)
        plot_pil = Image.open(buf).convert("RGB")
        plot_tensor = _pil_to_nhwc_tensor(plot_pil).unsqueeze(0)
        return (sims[idx], txts[idx], plot_tensor)

class MultiImageToTextSimilarity:
    @classmethod
    def INPUT_TYPES(cls):
        req = {"text": ("STRING",)}
        opt = {f"image{i}": ("IMAGE",) for i in range(1,9)}
        return {"required": req, "optional": opt}
    RETURN_TYPES = ("FLOAT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("highscore", "best_image", "plot")
    FUNCTION = "calc"
    CATEGORY = "multi/similarity"

    def calc(self, text, **images):
        pil_list = []
        for i in range(1,9):
            im = images.get(f"image{i}")
            if im is not None:
                pil_list.append(_to_pil(im))
        if not pil_list:
            return (0.0, None, None)
        i_feats = _encode_images(pil_list)
        t_feat = _encode_texts([text])[0:1].repeat(len(pil_list),1)
        sims = torch.cosine_similarity(i_feats, t_feat, dim=1).cpu().tolist()
        idx = int(torch.tensor(sims).argmax())
        best_pil = pil_list[idx]
        best_image = _pil_to_nhwc_tensor(best_pil).unsqueeze(0)
        fig, ax = plt.subplots(figsize=(len(sims)*1.0,4), dpi=150)
        ax.bar(range(len(sims)), sims)
        ax.set_xticks(range(len(sims))); ax.set_xticklabels([""]*len(sims))
        ax.set_ylabel("Cosine similarity"); ax.set_title("Image vs Text similarities")
        for i, pil in enumerate(pil_list):
            thumb = pil.resize((32,32))
            ab = AnnotationBbox(OffsetImage(thumb, zoom=1), (i,0),
                                xybox=(0,-20), xycoords='data',
                                boxcoords="offset points", frameon=False)
            ax.add_artist(ab)
        plt.tight_layout()
        buf = BytesIO(); fig.savefig(buf, format='PNG', dpi=150); plt.close(fig)
        buf.seek(0)
        plot_tensor = _pil_to_nhwc_tensor(Image.open(buf).convert("RGB")).unsqueeze(0)
        return (sims[idx], best_image, plot_tensor)

# ——— Aesthetic nodes ———

class AestheticScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    FUNCTION = "calc"
    CATEGORY = "aesthetic"

    def calc(self, image):
        pil = _to_pil(image)
        inputs = aest_preprocessor(images=pil, return_tensors="pt").pixel_values
        inputs = inputs.to(torch.bfloat16).to(device)
        with torch.inference_mode():
            logits = aest_model(inputs).logits.squeeze()
        return (float(logits.float().cpu().item()),)

class MultiAestheticScore:
    @classmethod
    def INPUT_TYPES(cls):
        req = {}
        opt = {f"image{i}": ("IMAGE",) for i in range(1,9)}
        return {"required": req, "optional": opt}
    RETURN_TYPES = ("FLOAT", "FLOAT", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("highscore", "lowscore", "best_image", "worst_image", "plot")
    FUNCTION = "calc"
    CATEGORY = "aesthetic"

    def calc(self, **images):
        # Collect connected images
        pil_list = []
        for i in range(1,9):
            im = images.get(f"image{i}")
            if im is not None:
                pil_list.append(_to_pil(im))
        if not pil_list:
            return (0.0, None, None)
        # Batch preprocess and score
        inputs = aest_preprocessor(images=pil_list, return_tensors="pt").pixel_values
        inputs = inputs.to(torch.bfloat16).to(device)
        with torch.inference_mode():
            logits = aest_model(inputs).logits.squeeze().float().cpu().numpy()
        scores = logits.tolist()
        idx = int(np.argmax(scores))
        highscore = float(scores[idx])
        best_pil = pil_list[idx]
        best_image = _pil_to_nhwc_tensor(best_pil).unsqueeze(0)
        # Plot
        fig, ax = plt.subplots(figsize=(len(scores)*1.0,4), dpi=150)
        ax.bar(range(len(scores)), scores)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(["" for _ in scores])
        ax.set_ylabel("Aesthetic score")
        ax.set_title("Aesthetic Scores Comparison")
        # Add thumbnails as x-labels
        for i, pil in enumerate(pil_list):
            thumb = pil.resize((32,32))
            img_box = OffsetImage(thumb, zoom=1)
            ab = AnnotationBbox(img_box, (i, 0), xybox=(0,-20), xycoords='data', boxcoords="offset points", frameon=False)
            ax.add_artist(ab)
        plt.tight_layout()
        buf = BytesIO(); fig.savefig(buf, format='PNG', dpi=150); plt.close(fig)
        buf.seek(0)
        plot_pil = Image.open(buf).convert("RGB")
        plot_tensor = _pil_to_nhwc_tensor(plot_pil).unsqueeze(0)
        worst_pil = pil_list[int(np.argmin(scores))]
        worst_image = _pil_to_nhwc_tensor(worst_pil).unsqueeze(0)
        lowscore = float(min(scores))
        return (highscore, lowscore, best_image, worst_image, plot_tensor)

# ——— Mappings ———

NODE_CLASS_MAPPINGS = {
    "TextSimilarity": TextSimilarity,
    "ImageSimilarity": ImageSimilarity,
    "MultiTextToImageSimilarity": MultiTextToImageSimilarity,
    "MultiImageToTextSimilarity": MultiImageToTextSimilarity,
    "AestheticScore": AestheticScore,
    "MultiAestheticScore": MultiAestheticScore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextSimilarity": "Text Similarity (CLIP)",
    "ImageSimilarity": "Image Similarity (CLIP)",
    "MultiTextToImageSimilarity": "Multi Text→Image Similarity",
    "MultiImageToTextSimilarity": "Multi Image→Text Similarity",
    "AestheticScore": "Aesthetic Score",
    "MultiAestheticScore": "Multi Aesthetic Comparison",
}
