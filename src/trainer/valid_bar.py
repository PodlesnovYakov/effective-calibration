import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

class SilentValidationBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True  
        return bar