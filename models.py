import torch
import torch.nn as nn
from utils import clip_clipping
from clip import clip
from transformers import ViTForImageClassification

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "SVHN": "This is a photo of a {}",
    "Resisc45": "This is a photo of a {}",
    "CLEVR": "This is a photo of {} objects",
    "LocMNIST": "This is a photo of {}",
    "ColourBiasedMNIST":"This is a photo of {}",
}

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class DecoderManual(nn.Module):
    def __init__(self, i_dim, src_dim, act=nn.GELU):
        super(DecoderManual, self).__init__()
        if i_dim: self.shared_feature = 1
        else:     self.shared_feature = 0
        if self.shared_feature:
            #! start from 7*7*16(784:16) or 7*7*32(1568:800) or 7*7*64(3,136:2368)
            if (src_dim % 49) != 0: raise ValueError('map dim must be devided with 7*7')
            self.p_trigger = torch.nn.Parameter(torch.Tensor(1, src_dim - i_dim))
            torch.nn.init.uniform_(self.p_trigger, a=0.0, b=0.1) # can be tuned
            src_c = src_dim // 49
        else:
            src_c = src_dim
        
        bias_flag = False
        body_seq = []
        
        if src_c >= 64:    g_c = 64
        else:              g_c = src_c
        body_seq              +=  [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=g_c),
                                    nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(64), act()]
        body_seq              +=  [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                                    nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(32), act()]
        body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                    nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(32), act()]
        body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                    nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(16), act()]
        body_seq              +=  [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]  

        self.body   = nn.Sequential(*body_seq)

    def forward(self, z):
        if self.shared_feature:
            N = z.shape[0]
            D = self.p_trigger.shape[1]
            p_trigger = self.p_trigger.repeat(N, 1)
            z_cube = torch.cat((z, p_trigger), dim=1)
            z_cube = z_cube.reshape(N, -1, 7, 7)
        else:
            return self.body(z)
        return self.body(z_cube)

class Coordinator(nn.Module):
    def __init__(self):
        super(Coordinator, self).__init__()
      
        self.backbone = 'vit-mae-base'
        act = nn.GELU #if args.TRAINER.BLACKVIP.ACT == 'gelu' else nn.ReLU
        src_dim = 1568
        
        z_dim = 768
        if self.backbone == 'vit-mae-base':   #! SSL-MAE VIT-B (n param: 86M)
            self.enc_pt = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")

        self.dec = DecoderManual(z_dim, src_dim, act=act)

    def forward(self, x):
        with torch.no_grad():
            if self.backbone == 'vit-mae-base':
                #! (N, 197, 768) => pick [CLS] => (N, 768)
                out = self.enc_pt(x, output_hidden_states=True)
                z = out.hidden_states[-1][:,0,:]
        
        wrap = self.dec(z)
        return wrap, z

class CustomCLIP(nn.Module):
    '''editted for visual prompting'''
    def __init__(self):
        super().__init__()
        classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        clip_model = load_clip_to_cpu("ViT-B/16").float()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.p_eps = 1.0

        temp = CUSTOM_TEMPLATES["SVHN"]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        self.n_classes = len(classnames)
        self.classnames = classnames

        print(f"Text Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features.cuda()
        self.coordinator = Coordinator()
 
    def forward(self, image):
        prompt, _  = self.coordinator(image.type(self.dtype))
        prompted_images = clip_clipping(image + self.p_eps * prompt)

        image_features = self.image_encoder(prompted_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()

        return logits