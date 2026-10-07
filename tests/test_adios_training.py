"""CPU tests with a fake ADIOS reader that enforces hyperslab reads."""
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from pyCAESAR.adios_dataset import AdiosPatchDataset
from pyCAESAR.models.utils import convert_args
from pyCAESAR.train_vae3d import test_adios_patches


class Reader:
    values = {}
    reads = []
    def __init__(self, path): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def close(self): pass
    def available_variables(self):
        return {name: {'AvailableStepsCount': str(len(steps)), 'Type':'float',
                      'Shape': ', '.join(map(str, steps[0].shape))}
                for name, steps in self.values.items()}
    def read(self, name, start, count, step_selection):
        self.reads.append((name, start, count, step_selection))
        return self.values[name][step_selection[0]][tuple(slice(a,a+n) for a,n in zip(start,count))]


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return {'output':x, 'frame_bit':torch.ones(len(x))}


class AdiosTests(unittest.TestCase):
    def setUp(self):
        Reader.reads = []
        self.fake = patch.dict(sys.modules, {'adios2':types.SimpleNamespace(FileReader=Reader)})
        self.fake.start()
    def tearDown(self): self.fake.stop()
    def args(self, train=False):
        return dict(data_path='fake.bp', data_format='adios', adios={'lazy':True,'variables':'auto',
                    'axes':'auto','steps_as_time':'auto'}, n_frame=16, n_overlap=0,
                    train=train, train_size=8, test_size=[8,8])
    def test_single_step_ranks_and_bounded_reads(self):
        x = np.arange(19*11*13,dtype=np.float32).reshape(19,11,13)
        for value in (x,x[None],x[None,None]):
            Reader.values={'field':[value]}
            dataset=AdiosPatchDataset(self.args())
            self.assertEqual(len(dataset),8)
            nrmse,bpp=test_adios_patches(IdentityModel(),DataLoader(dataset,batch_size=2),torch.device('cpu'))
            self.assertLess(nrmse,1e-6)
            self.assertGreater(bpp,0)
            self.assertTrue(all(count[-1]<=8 and count[-2]<=8 for _,_,count,_ in Reader.reads))
    def test_multistep_and_multiple_fields(self):
        x=np.arange(2*11*13,dtype=np.float32).reshape(2,11,13)
        Reader.values={'a':[x+i for i in range(19)],'b':[x+i+1 for i in range(19)]}
        dataset=AdiosPatchDataset(self.args())
        self.assertEqual(len(dataset),32)
        self.assertEqual(dataset[0]['input'].shape,(1,16,8,8))
        self.assertEqual(len(Reader.reads),16)
        self.assertEqual(dataset[4]['valid_shape'].tolist(),[3,8,8])
    def test_constant_padding(self):
        Reader.values={'field':[np.ones((3,4,5),dtype=np.float32)]}
        dataset=AdiosPatchDataset(self.args())
        sample=dataset[0]
        self.assertEqual(sample['valid_shape'].tolist(),[3,4,5])
        self.assertTrue(torch.isfinite(sample['input']).all())
        self.assertEqual(sample['input'].abs().sum(),0)
    def test_config_sizes(self):
        for size in (128,256,512):
            args=types.SimpleNamespace(config='examples/config_adios_experiment.yaml',
                train_set='re3200,hurricane,input300',test_set='input300',spatial_size=size)
            self.assertEqual(len(convert_args(args)),3)
            self.assertTrue(all(v['train_size']==size for v in convert_args(args).values()))
            self.assertEqual(convert_args(args,False)['input300']['test_size'],[size,size])

if __name__=='__main__': unittest.main()
