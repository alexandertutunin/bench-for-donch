import random
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

from matplotlib import rcParams
rcParams['axes.formatter.useoffset'] = False
rcParams['text.usetex'] = False


class Monitor:
    def __init__(self, numshow = 60, long_spot_type = "running"):
        '''
        **Parameters**:\n
            numshow - how many latest epochs to show.

            long_spot_type: [ "fixed" | "running"] - type of long-time spot value. Type "fixed" ensures
            long-time spot loss will be calculated w.r.t. global primary value (epoch 0). 
            Type "running" will ignore global primary value (epoch 0) and long-time spot loss will 
            be calculated w.r.t. the most old value within numshow latest epochs values.
        '''
        if hasattr(tqdm.tqdm, '_instances'):
            [*map(tqdm.tqdm._decr_instances, list(tqdm.tqdm._instances))]

        self.numshow = numshow
        self.lst = long_spot_type

        self.train_component_names = None
        self.train_component_values = None

        self.train_loss_curve = []
        self.train_bce = []
        self.train_sharp = []
        self.train_nf = []

        self.test_component_names = None
        self.test_component_values = None

        self.test_loss_curve = []
        self.test_bce = []
        self.test_sharp = []
        self.test_nf = []

        self.gradnorms = None
        self.clipped_gradnorms = None

        self.lr_curve = []

    def add_lr(self, lr):
        self.lr_curve.append(lr)

    def add_gradnorms(self, gradnorms):
        if self.gradnorms is not None:
            self.gradnorms = torch.cat([self.gradnorms, torch.mean(gradnorms,dim=0).unsqueeze(0)], dim=0)
        else:
           print("is none")
           self.gradnorms = torch.mean(gradnorms,dim=0).unsqueeze(0)
        #    self.gradnorms = self.gradnorms.unsqueeze(0)

    def add_clipped_gradnorms(self, clipped_gradnorms):
        if self.clipped_gradnorms is not None:
            self.clipped_gradnorms = torch.cat([self.clipped_gradnorms, torch.mean(clipped_gradnorms,dim=0).unsqueeze(0)], dim=0)
        else:
           print("is none")
           self.clipped_gradnorms = torch.mean(clipped_gradnorms,dim=0).unsqueeze(0)
        #    self.gradnorms = self.gradnorms.unsqueeze(0)
        
    def add_train_loss(self, loss):
        self.train_loss_curve.append(loss)
    
    def add_train_loss_components(self, bce, sharp, nf):
        self.train_bce.append(bce)
        self.train_sharp.append(sharp)
        self.train_nf.append(nf)
    
    def add_named_loss_components(self, components: dict, source="train"):

        if source == "train":
            if not self.train_component_names: self.train_component_names = components["names"]
            else: pass

            if self.train_component_values is not None:
                self.train_component_values = torch.cat([self.train_component_values, components["values"].unsqueeze(1)], dim=-1)
            else:
                self.train_component_values = components["values"].unsqueeze(1)
        
        elif source == "test":
            if not self.test_component_names: self.test_component_names = components["names"]
            else: pass

            if self.test_component_values  is not None:
                self.test_component_values = torch.cat([self.test_component_values, components["values"].unsqueeze(1)], dim=-1)
            else:
                self.test_component_values = components["values"].unsqueeze(1)

    def add_test_loss(self, loss):
        self.test_loss_curve.append(loss)

    def add_test_loss_components(self, bce, sharp, nf):
        self.test_bce.append(bce)
        self.test_sharp.append(sharp)
        self.test_nf.append(nf)
    
    def skip(self, n):
        
        old_size = len(self.train_loss_curve)
        self.train_loss_curve = self.train_loss_curve[:-n:]
        self.train_bce = self.train_bce[:-n:]
        self.train_sharp = self.train_sharp[:-n:]
        self.train_nf = self.train_nf[:-n:]

        self.test_loss_curve = self.test_loss_curve[:-n:]
        self.test_bce = self.test_bce[:-n:]
        self.test_sharp = self.test_sharp[:-n:]
        self.test_nf = self.test_nf[:-n:]

        print(f"{n} epochs skipped: [{old_size}] -> [{len(self.train_loss_curve)}]")

    def show(self):
        display.clear_output(wait=True)
        plt.figure()
        
        plt.plot(self.train_loss_curve, label="Train Loss", marker="o")
        plt.plot(self.test_loss_curve, label="Test Loss", marker="s")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Test Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_gradnorms_by_param(self, param_names):
        """
        Плотит нормы градиентов по каждому параметру модели в отдельных сабплотах.
        
        Parameters:
            param_names (list of str): список имен параметров модели (в том же порядке, как и в self.gradnorms)
        """

        if self.gradnorms is None:
            print("No gradient norms to plot.")
            return
        
        num_params = len(param_names)
        

        cols = 3
        rows = int(np.ceil(num_params / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        fig.suptitle("Gradient Norms per Parameter", fontsize=16)

        for idx, name in enumerate(param_names):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]
            
            ax.plot(self.gradnorms[:, idx].cpu().numpy(), marker='o')
            ax.set_title(name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Grad Norm")
            ax.grid(True)

        # Отключить пустые сабплоты
        for idx in range(num_params, rows * cols):
            fig.delaxes(axes[idx // cols][idx % cols])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_clipped_gradnorms_by_param(self, param_names):
            """
            Плотит нормы градиентов по каждому параметру модели в отдельных сабплотах.
            
            Parameters:
                param_names (list of str): список имен параметров модели (в том же порядке, как и в self.gradnorms)
            """

            if self.clipped_gradnorms is None:
                print("No gradient norms to plot.")
                return
            
            num_params = len(param_names)
            

            cols = 3
            rows = int(np.ceil(num_params / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
            fig.suptitle("Clipped Gradient Norms per Parameter", fontsize=16)

            for idx, name in enumerate(param_names):
                row = idx // cols
                col = idx % cols
                ax = axes[row][col]
                
                ax.plot(self.clipped_gradnorms[:, idx].cpu().numpy(), marker='o', color='orange')
                ax.set_title(name)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Grad Norm")
                ax.grid(True)

            # Отключить пустые сабплоты
            for idx in range(num_params, rows * cols):
                fig.delaxes(axes[idx // cols][idx % cols])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def deep_show(self):
        display.clear_output(wait=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        
        if len(self.train_loss_curve) < self.numshow or self.numshow == -1:
            indexing = np.arange(len(self.train_loss_curve)) 
        else:
            indexing = np.arange(len(self.train_loss_curve) - self.numshow, len(self.train_loss_curve))

        axes[0].plot(np.array(self.train_loss_curve)[indexing], label="Total Loss",     marker="o", color='blue')
        axes[0].plot(np.array(self.train_bce)[indexing],        label="BCE",            marker="x", color='orange')
        axes[0].plot(np.array(self.train_sharp)[indexing],      label="Penalty Sharp",  marker="x", color='green')
        axes[0].plot(np.array(self.train_nf)[indexing],         label="Penalty Zeros",  marker="x", color='gray')

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        anchor = np.array(self.train_loss_curve)[indexing][0] if self.lst == "running" else self.train_loss_curve[0]
        spot = (self.train_loss_curve[-1] - self.train_loss_curve[-2]) if len(self.train_loss_curve) > 1 else 0.0
        long = (self.train_loss_curve[-1] - anchor) if len(self.train_loss_curve) > 1 else 0.0

        plus_symbol = {True: "+", False: ""}

        axes[0].set_title(f"Training Loss Curve | Spot: {plus_symbol[spot>0]}{spot:.5f} | Long: {plus_symbol[long>0]}{long:.5f}")
        axes[0].legend()
        
        axes[0].set_ylim((-0.01, 1.1*max(np.array(self.train_loss_curve)[indexing])))
        axes[0].grid(True)

        axes[1].plot(np.array(self.test_loss_curve)[indexing],  label="Total Loss",     marker="o",     color='blue')
        axes[1].plot(np.array(self.test_bce)[indexing],         label="BCE",            marker="x",     color='orange')
        axes[1].plot(np.array(self.test_sharp)[indexing],       label="Penalty Sharp",  marker="x",     color='green')
        axes[1].plot(np.array(self.test_nf)[indexing],          label="Penalty Zeros",  marker="x",     color='gray')

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        anchor = np.array(self.test_loss_curve)[indexing][0] if self.lst == "running" else self.test_loss_curve[0]
        spot = (self.test_loss_curve[-1] - self.test_loss_curve[-2]) if len(self.test_loss_curve) > 1 else 0.0
        long = (self.test_loss_curve[-1] - anchor) if len(self.test_loss_curve) > 1 else 0.0
        
        axes[1].set_title(f"Test Loss Curve | Spot: {plus_symbol[spot>0]}{spot:.5f} | Long: {plus_symbol[long>0]}{long:.5f}")
        axes[1].legend()
        axes[1].set_ylim((-0.01, 1.1*max(np.array(self.test_loss_curve)[indexing])))
        axes[1].grid(True)

        plt.tight_layout()
        display.display(plt.gcf())
        plt.close()