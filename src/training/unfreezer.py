import torch
class ProgressiveUnfreezer:
    """ man I should make more docstrings like this, this is nuts. 
    Progressive unfreezing for Densenet121 
    usage:
        unfreezer = ProgresiveUnfreezer(model)

        for epoch in range(num_epochs):
            if freezer.should_update(epochs):
                optimizer = unfreezer.update(epoch, optmiizer)
    """

    def __init__(self, model, strategy="Progressive"):
        self.model = model
        self.strategy = strategy
        self.current_stage = 0
        self.stage_epochs = {0: 5, 1: 10, 2: 20} # could not tell you waht is happening here. 

        self._freeze_all()
        self._unfreeze_classifier()

    def _freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False


    def _unfreeze_last_block(self):
        for name, p in self.model.features.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                p.requires_grad = True

    def _unfreeze_classifier(self):
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    

    # turns out this was faaar to aggressive. trying again. 
    # def _unfreeze_denseblock(self, block_num):
    #     # unfreeze a specific dense block (1-4)
    #     block_name = f'denseblock{block_num}'
    #     for name, param in self.model.features.named_parameters():
    #         if block_name in name:
    #             param.requires_grad = True

    # def _unfreeze_transition(self, trans_num):
    #     trans_name = f'transition{trans_num}'
    #     for name, param in self.model.features.named_parameters():
    #         if trans_name in name:
    #             param.requires_grad = True

    # ok how exactly does htis function work I no understand good 
    def get_stage_for_epoch(self, epoch):
        "Determin which stage of unfreezing we should be given the epoch"
        cumulative = 0
        for stage, ep in self.stage_epochs.items():
            cumulative += ep
            if epoch < cumulative:
                return stage
            
        return max(self.stage_epochs.keys())
    
    def should_update(self, epoch):
        # check if we need to change unfreezing stage
        new_stage = self.get_stage_for_epoch(epoch)
        return new_stage != self.current_stage
    
    def update(self, epoch, old_optimizer=None):
            new_stage = self.get_stage_for_epoch(epoch)
            if new_stage == self.current_stage:
                return old_optimizer
            
            if new_stage == 0:
                lr = 1e-3
                desc = "Classifier only"
            elif new_stage == 1:
                self._unfreeze_last_block()
                lr = 1e-4
                desc = "Classifier + DenseBlock4"
            elif new_stage == 2:
                # NEVER unfreeze everything—just add transition3
                for name, p in self.model.features.named_parameters():
                    if 'transition3' in name or 'denseblock3' in name:
                        p.requires_grad = True
                lr = 5e-5
                desc = "Classifier + DenseBlocks 3-4"
            
            self.current_stage = new_stage
            # trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            # total = sum(p.numel() for p in self.model.parameters())
            # print(f"  {desc} | LR: {lr} | Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")
            
            if old_optimizer:
                return torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=lr, weight_decay=1e-4
                )
            return None

        