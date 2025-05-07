import numpy as np
import torch
torch.manual_seed(1337)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import pandas as pd
import tqdm

from metrics import double_metric
import time

class Decomposer():
    '''
    Base Decoder
    Provides wavedec for a given signal, concatenates dec. levels and flattens them 
    into a single vector dropping NAs. 
    '''
    def __init__(self, maxcurrent = 500):
        self.max_current = maxcurrent
        pass
    
    @staticmethod
    def _toAmperes(data):
        return data/1e3
    
    def _normalize_by_current(self, data):
        return data / (self.max_current * np.sqrt(2))
    
    def normalize(self, batch: list):
        '''
        Takes raw batch and returns it normalized in amperes
        '''
        result = []
        for data in batch:
            data = self._toAmperes(data)
            result.append(self._normalize_by_current(data))
        return result

class WindowSamplerTorch:
    def __init__(self, data, labels, wsize=80, stride=4, idx_start=0, padding_mode="same", smp_per_period = 80):
        assert data.dim() == 2, "Data must have shape (num_units, unit_len)"
        assert labels.dim() == 2, "Labels must have shape (num_units, unit_len)"

        self.data = data
        self.labels = labels
        self.wsize = wsize
        self.stride = stride
        self.idx_start = idx_start
        self.padding_mode = padding_mode

        self.smp = smp_per_period

        self._pad()

    def _pad(self):
        
        if self.padding_mode == "same":
            pad_values = self.data[:, :self.wsize]      #takes first wsize elems
            pad_labels = self.labels[:, :self.wsize]    #takes first wsize elems
        elif self.padding_mode == "zeros":
            pad_values = torch.zeros((self.data.shape[0], self.wsize), dtype=self.data.dtype)
            pad_labels = torch.zeros((self.labels.shape[0], self.wsize), dtype=self.labels.dtype)
        else:
            raise ValueError("Unsupported padding mode")

        n = (self.wsize - 1) // self.smp + 1
        
        pad_values = pad_values.repeat(1, n)[:, -self.wsize+1:] #takes last wsize-1 elems
        pad_labels = pad_labels.repeat(1, n)[:, -self.wsize+1:] #takes last wsize-1 elems

        self.data = torch.cat([pad_values, self.data], dim=1)
        self.labels = torch.cat([pad_labels, self.labels], dim=1)

    def get_all_windows(self):
        
        windows = self.data[:, self.idx_start:].unfold(1, self.wsize, self.stride)  # (num_units, num_windows, wsize)
        labels = self.labels[:, self.idx_start:].unfold(1, self.wsize, self.stride)[:, :, -1]
        
        return windows, labels

    def __getitem__(self, idx):
        """ Возвращает окна для конкретного индекса вдоль unit_len """
        return self.get_all_windows()[0][:, idx, :], self.get_all_windows()[1][:, idx]

    def __len__(self):
        """ Количество окон, доступных для семплирования """
        return (self.data.shape[1] - self.idx_start - self.wsize) // self.stride + 1

class DatasetWindowed(Dataset):
    def __init__(self, data, labels, wsize=80, stride=4, start_idx=0, padding_mode='same', device="cpu"):
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.uint8).to(device)
        self.wsize = wsize
        self.stride = stride
        self.start_idx = start_idx
        self.padding_mode = padding_mode

        
        self.sampler = WindowSamplerTorch(
            self.data, self.labels, wsize=self.wsize, stride=self.stride, idx_start=self.start_idx, padding_mode=self.padding_mode
        )
        
        self.all_windows, self.all_labels = self.sampler.get_all_windows()  # (num_units, num_windows, wsize), (num_units, num_windows)
        # self.all_windows.to(device)
        # self.all_windows.to(device)

    def to(self, device):
        self.all_windows = self.all_windows.to(device)
        self.all_labels = self.all_labels.to(device)
        pass

    def cpu(self):
        self.to("cpu")
        pass


    def __len__(self):
        return len(self.all_windows)  

    def __getitem__(self, index):
        """
        - data: (num_windows, wsize)
        - labels: (num_windows,)
        """
        unit_windows = self.all_windows[index]  # (num_windows, wsize)
        unit_labels = self.all_labels[index]  # (num_windows,)

        return unit_windows.to(dtype=torch.float32), unit_labels.to(dtype=torch.int8)

class Timer():

    def __init__(self, timing = 'us'):
        '''
        Set `timing` to define by what unit time to be showed \n s - seconds,\n ms - 1e-3 seconds,\n us - 1e-6 seconds,\n ns - 1e-9 seconds)
        '''
        self.timing = timing
        self.time_coef = {'s': 1.0, 'ms': 1e3, 'us': 1e6, 'ns': 1e9}[timing]
        
        self.ticks = {}
        self.tucks = {}
        self.is_tick = {}
        self.is_tuck = {}

        self.result = {}

        self.statistics = None

        pass

    def _get_ts(self):
        return round(time.time_ns() / (1e9/self.time_coef), 9)
    
    def tick(self, id = "common"):
        
        # if id != "common":
        #     self.is_tick[id] = True
        # else: pass
        if id not in self.ticks.keys():
            self.ticks[id] = [self._get_ts()]
        else:
            self.ticks[id].append(self._get_ts())
        
    def tuck(self, id = "common"):
        
        # if id != "common" and self.is_tick:
        #     self.flags[id] = "self"

        if id not in self.tucks.keys():
            self.tucks[id] = [self._get_ts()]
        else:
            self.tucks[id].append(self._get_ts())

    def _get_result(self):
        result = {}
        for key in self.tucks.keys():
            if len(self.ticks[key]) != len(self.tucks[key]):
                raise IndexError(f"Not equal number of ticks and tucks for id: {key}. Check if you have unclosed ticks or tucks")
            else:
                
                result[key] = (np.array(self.tucks[key]) - np.array(self.ticks[key])).tolist()
        return result

    def show(self):
        result = self._get_result()
        columns=[key + f" [{self.timing}]" for key in result.keys()]
        result = pd.DataFrame.from_dict(result, orient='index').T
        result.columns = columns
        return result
    
    def stats(self):
        indices = ["min", "avg", "max"]
        resdf = self.show()
        funcmin = resdf.min()
        funcavg = resdf.mean()
        funcmax = resdf.max()
        statistics = pd.DataFrame()
        for index, func in zip(indices, [funcmin, funcavg, funcmax]):
            statistics = pd.concat(
                [
                    statistics,
                    pd.DataFrame(func, columns=[index]).T
                ]
            )
            
        return statistics

def make_distant_old(X, scaling=1.0):
    X = X**2
    sigmoidal = 2 / (1 + np.exp(-scaling*(abs(X) - 1.0))) * np.sign(X)
    # sigmoidal = np.where(abs(X) <= 1.0, X, sigmoidal) 
    return sigmoidal

def train_epoch(
        model: nn.Module, 
        loader: torch.utils.data.DataLoader, 
        criterion: nn.Module,
        optimizer,
        scheduler = None, 
        tloss:float = 1.0, 
        verbose:int = 1,
        device="cuda",
        grad_clipping = None,
        ):
    
    model.train()
    
    train_loss = 0.0
    train_score = 0.0
    faulty_f1, nonfaulty_f1 = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
    train_components = {"names": [], "values": []}

    grad_norms = torch.Tensor([torch.nan]*len(list(model.parameters()))).to(device)
    grad_norms = grad_norms.unsqueeze(0)

    clipped_grad_norms = torch.Tensor([torch.nan]*len(list(model.parameters()))).to(device)
    clipped_grad_norms = clipped_grad_norms.unsqueeze(0)

    # print(f"grad norms empty size: {grad_norms.size()}")
    # param_grad_norm_min = torch.Tensor([]).to(device)
    # params_grad_avg = torch.Tensor([]).to(device)
    # params_grad_max = torch.Tensor([]).to(device)

    itr_obj = tqdm.tqdm(loader, total=len(loader), desc='   Training') if verbose else loader

    for batch_data, batch_target in itr_obj:
        
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        
        if torch.isnan(batch_data).any():
            raise Warning("NaN in input data")
        if torch.isinf(batch_data).any():
            raise Warning("Inf in input data")

        # Forward pass
        out_logits = model(batch_data) #logits

        invalid_model = False
        if torch.isnan(out_logits).any():
            invalid_model = True
            raise Warning(f"ModelOut: NaN in model output")
        if torch.isinf(out_logits).any():
            invalid_model = True
            raise Warning(f"ModelOut: Inf in model output")
   

        loss = criterion(out_logits, batch_target.float(), t=tloss) #as logits, sigmoid inside
        components = criterion.named_loss_components()
        # components = {"names": ["bce", ...], "values": Tensor([1.248, ...])}

        if torch.isnan(loss).any():
            raise Warning("NaN in Loss")
        if torch.isinf(loss).any():
            raise Warning("Inf in Loss")

        # print(f"Loss on Batch: {loss.item():.3f}")
        
        # score = custom_metric_torch_batch(y_true=batch_target, y_pred=(torch.sigmoid(out_logits) > 0.5), beta=1.0)
        mf, mnf = double_metric(y_true=batch_target, y_pred=(torch.sigmoid(out_logits) > 0.5), beta=1.0)
        faulty_f1 = torch.cat([faulty_f1, mf])
        nonfaulty_f1 = torch.cat([nonfaulty_f1, mnf])

        # train_score += score.mean()
        train_loss += loss.item()
        train_components["names"] = components["names"]
        if train_components["values"]:
            train_components["values"] += components["values"]
        else:
            train_components["values"] = components["values"]


        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        
        # grad_norms = torch.zeros(size=(len(list(model.parameters()))),dtype=torch.float32).to(device)
        # grad_norms.squeeze(0) #add new dim=0
        norms = []
        for name, param in model.named_parameters():
            norms.append(torch.norm(param.grad))

            if torch.isnan(param).any():
                invalid_model = True
                print(f"Params after Back Propagation: NaN in param {name}")
            if torch.isinf(param).any():
                invalid_model = True
                print(f"Params after Back Propagation: Inf in param {name}")
              
            if torch.isnan(param.grad).any():
                invalid_model = True
                print(f"Gradients after Back Propagation: NaN in grad of {name}")
            if torch.isinf(param.grad).any():
                invalid_model = True
                print(f"Gradients after Back Propagation: Inf in grad of {name}")           
        
        # grad_norms += torch.Tensor(norms) # size [n_params,], avg grad norm over whole epoch for each param
        # print(f"len norms: {len(norms)}")
        norms = torch.Tensor(norms).to(device)
        norms = norms.unsqueeze(0)
        grad_norms = torch.cat([grad_norms, norms], dim=0) #size [batch_size*len(loader), num_params]
        # print(f"tensor norms size: {norms.size()}")

        if grad_clipping is not None:

            torch.nn.utils.clip_grad_norm_(model.rnn.parameters(), max_norm=grad_clipping)

        norms = []
        for name, param in model.named_parameters():
            norms.append(torch.norm(param.grad))
            
            if torch.isnan(param).any():
                invalid_model = True
                print(f"Params after Grad Clipping: NaN in param {name}")
            if torch.isinf(param).any():
                invalid_model = True
                print(f"Params after Grad Clipping: Inf in param {name}")
              
            if torch.isnan(param.grad).any():
                invalid_model = True
                print(f"Gradients after Grad Clipping: NaN in grad of {name}")
            if torch.isinf(param.grad).any():
                invalid_model = True
                print(f"Gradients after Grad Clipping: Inf in grad of {name}")
        
        norms = torch.Tensor(norms).to(device)
        norms = norms.unsqueeze(0)
        clipped_grad_norms = torch.cat([clipped_grad_norms, norms], dim=0)
        
        
        optimizer.step()

        if invalid_model: raise InterruptedError("Invalid Model: loop broken")


    if scheduler is not None: 
        scheduler.step()
        # print(f"scheduler did step")

    
    train_score = faulty_f1.mean() * nonfaulty_f1.mean()
    # print(f"Train Score: {train_score*100:.3f}%")

    train_loss /= len(loader)
    train_score /= len(loader)
    train_components["values"] /= len(loader)
    # grad_norms /= len(loader)

    return train_loss, train_score*100, train_components, grad_norms[1:, :], clipped_grad_norms[1:, :]

def test_epoch(
        model: nn.Module, 
        loader: torch.utils.data.DataLoader, 
        criterion: nn.Module, 
        tloss:float = 1.0, 
        verbose:int = 1,
        device = "cuda"
        ):
    
    model.eval()
    with torch.no_grad():
    
        test_loss = 0.0
        test_score = 0.0
        faulty_f1, nonfaulty_f1 = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        test_components = {"names": [], "values": []}

        itr_obj = tqdm.tqdm(loader, total=len(loader), desc='   Evaluation') if verbose else loader

        for batch_data, batch_target in itr_obj:
            
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            
            if torch.isnan(batch_data).any():
                raise Warning("NaN in input data")
            if torch.isinf(batch_data).any():
                raise Warning("Inf in input data")

            # Forward pass
            out_logits = model(batch_data) #logits

            invalid_model = False
            if torch.isnan(out_logits).any():
                invalid_model = True
                raise Warning(f"ModelOut: NaN in model output")
            if torch.isinf(out_logits).any():
                invalid_model = True
                raise Warning(f"ModelOut: Inf in model output")
    
            loss = criterion(out_logits, batch_target.float(), t=tloss) #as logits, sigmoid inside
            components = criterion.named_loss_components()
            # components = {"names": ["bce", ...], "values": Tensor([1.248, ...])}

            if torch.isnan(loss).any():
                raise Warning("NaN in Loss")
            if torch.isinf(loss).any():
                raise Warning("Inf in Loss")

            # print(f"Loss on Batch: {loss.item():.3f}")
            
            # score = custom_metric_torch_batch(y_true=batch_target, y_pred=(torch.sigmoid(out_logits) > 0.5), beta=1.0)
            mf, mnf = double_metric(y_true=batch_target, y_pred=(torch.sigmoid(out_logits) > 0.5), beta=1.0)
            faulty_f1 = torch.cat([faulty_f1, mf])
            nonfaulty_f1 = torch.cat([nonfaulty_f1, mnf])

            # test_score += score.mean()
            test_loss += loss.item()
            test_components["names"] = components["names"]
            if test_components["values"]:
                test_components["values"] += components["values"]
            else:
                test_components["values"] = components["values"]

        test_score = mf.mean() * mnf.mean()
        # print(f"Train Score: {test_score*100:.3f}%")

        test_loss /= len(loader)
        test_score /= len(loader)
        test_components["values"] /= len(loader)

        if invalid_model: raise InterruptedError("Invalid Model: loop broken")
        
    return test_loss, test_score*100, test_components 

class StructuredBCELoss(torch.nn.Module):
    def __init__(self, pos_weight = 1.0, reg_sharpness=1.0, reg_nonfaulty = 1.0):
        super().__init__()
        self.reg_sharpness = reg_sharpness
        self.reg_nonfaulty = reg_nonfaulty

        self.weight = pos_weight

        self.penalty_sharpness = None
        self.penalty_nonfaulty = None
        self.BCEvalue = None

    def forward(self, pred_logits, y_true, t=0.0):
        """
        pred_logits: (batch_size, sequence_length) - Сырые выходы модели (логиты)
        y_true: (batch_size, sequence_length) - Истинные метки (0 или 1)
        """

        t = torch.tensor(t, dtype=torch.float32, device=pred_logits.device)
        weight = torch.as_tensor([self.weight], dtype=torch.float32, device=pred_logits.device)

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, 
            y_true, 
            weight=weight, 
            reduction="mean"
            )
        pred_probs = torch.sigmoid(pred_logits)

        
        mask = y_true  
        diff = torch.zeros_like(mask)
        diff[:, 1:] = pred_probs[:, 1:] - pred_probs[:, :-1]

    
        #penalty for sharp transition 1 -> 0 excepting 2 allowed sharp transitions
        penalty_sharpness = self.reg_sharpness * t * torch.sum(mask * torch.clamp(-diff, min=0) ** 2)

        #penalty for correctly predict non-faulty unit - Loss for non-faulty should not be 0.0
        penalty_nonfaulty = self.reg_nonfaulty * torch.sum((torch.sum(y_true, dim=-1) == 0))

        # print(f"bce:                {bce_loss}")
        # print(f"penalty_sharpness:  {penalty_sharpness}")
        # print(f"penalty_nonfaulty:  {penalty_nonfaulty}\n")
        total_loss = bce_loss +  penalty_sharpness + penalty_nonfaulty
        
        self.penalty_sharpness = penalty_sharpness.detach()
        self.penalty_nonfaulty = penalty_nonfaulty.detach()
        self.BCEvalue = bce_loss.detach()
        
        return total_loss
    
    def named_loss_components(self):
        names = ["BCE"]
        values = [self.BCEvalue]

        if self.reg_sharpness != 0.0:
            names.append("Sharpness")
            values.append(self.penalty_sharpness)

        if self.reg_nonfaulty != 0.0:
            names.append("NonFaulty")
            values.append(self.penalty_nonfaulty)
        
        loss_components = {
            "names": names,
            "values": torch.stack(values) 
        }

        return loss_components