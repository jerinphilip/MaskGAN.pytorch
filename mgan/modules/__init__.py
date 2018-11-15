
from .mgan_trainer import MGANTrainer
from .mle_trainer import MLETrainer

def build_trainer(tag, args, task):
    if tag == 'MLE':
        trainer = MLETrainer(args, task)
        return trainer

    elif tag == 'MGAN':
        trainer = MGANTrainer(args, task)
        return trainer
    
    else:
        raise Exception("Unknown tag")


