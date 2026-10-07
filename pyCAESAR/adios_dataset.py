"""Read only one training/evaluation patch at a time from an ADIOS BP file."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from pyCAESAR.dataset import BaseDataset
from pyCAESAR.data_io import canonicalize


class AdiosPatchDataset(BaseDataset):
    streaming = True

    def __init__(self, args):
        super().__init__(args)
        from adios2 import FileReader
        self.reader = None
        self.fields = []
        self.size = self.train_size if self.train_mode else self.test_size[0]
        if not self.train_mode and self.test_size[0] != self.test_size[1]:
            raise ValueError('ADIOS experiment requires square test blocks')
        if self.n_overlap:
            raise ValueError('ADIOS experiment uses n_overlap: 0')
        if not self.inst_norm:
            raise ValueError('ADIOS patch input requires inst_norm: true')
        with FileReader(self.data_path) as reader:
            metadata = reader.available_variables()
            names = self.adios_config.get('variables', 'auto')
            if names == 'auto':
                names = [name for name, info in metadata.items()
                         if info.get('Type') in ('float', 'double')
                         and 3 <= len(self._shape(info)) <= 5]
            elif isinstance(names, str):
                names = [names]
            if not names:
                raise ValueError(f'No floating-point 3D–5D variables in {self.data_path}; configure adios.variables')
            for name in names:
                if name not in metadata:
                    raise ValueError(f'{name!r} missing; available variables: {list(metadata)}')
                info = metadata[name]
                shape = self._shape(info)
                total_steps = int(info['AvailableStepsCount'])
                steps_time = self.adios_config.get('steps_as_time', 'auto')
                steps_time = total_steps > 1 if steps_time == 'auto' else steps_time
                axes = self.adios_config.get('axes', 'auto')
                if axes == 'auto':
                    layouts = ({2: ['height', 'width'], 3: ['section', 'height', 'width'],
                                4: ['variable', 'section', 'height', 'width']} if steps_time else
                               {3: ['time', 'height', 'width'], 4: ['section', 'time', 'height', 'width'],
                                5: ['variable', 'section', 'time', 'height', 'width']})
                    if len(shape) not in layouts:
                        raise ValueError(f'Cannot infer axes for {name}: {shape}; set adios.axes')
                    axes = layouts[len(shape)]
                axes = list(axes)
                canonicalize(np.empty([1] * (len(shape) + (steps_time and 'time' not in axes))),
                             (['time'] + axes) if steps_time and 'time' not in axes else axes)
                if len(axes) != len(shape) or (steps_time and 'time' in axes):
                    raise ValueError('For steps_as_time, omit time from per-step axes')
                dimensions = dict(zip(axes, shape))
                step_start, step_stop = self.adios_config.get('step_range', [0, total_steps if steps_time else 1])
                if not 0 <= step_start < step_stop <= total_steps or (not steps_time and step_stop-step_start != 1):
                    raise ValueError('Invalid adios.step_range')
                frames = step_stop - step_start if steps_time else dimensions['time']
                t0, t1 = self.frame_range or (0, frames)
                s0, s1 = self.section_range or (0, dimensions.get('section', 1))
                if not 0 <= t0 < t1 <= frames or not 0 <= s0 < s1 <= dimensions.get('section', 1):
                    raise ValueError(f'Invalid frame/section selection for {name}')
                variables = self.variable_idx if self.variable_idx is not None else range(dimensions.get('variable', 1))
                variables = np.atleast_1d(variables).tolist() if isinstance(variables, (int, list)) else list(variables)
                for variable in variables:
                    if not 0 <= variable < dimensions.get('variable', 1):
                        raise ValueError(f'Invalid variable_idx for {name}')
                    h, w = dimensions['height'], dimensions['width']
                    nt = math.ceil((t1-t0)/self.n_frame)
                    nh, nw = math.ceil(h/self.size), math.ceil(w/self.size)
                    self.fields.append(dict(name=name, axes=axes, shape=shape, steps_time=steps_time,
                                            step_start=step_start, variable=variable, s0=s0, s1=s1,
                                            t0=t0, t1=t1, h=h, w=w, nt=nt, nh=nh, nw=nw,
                                            length=(s1-s0)*nt*nh*nw))
                print(f'ADIOS {name}: shape={shape}, axes={axes}, steps_as_time={steps_time}', flush=True)
        self.dataset_length = sum(field['length'] for field in self.fields)
        self.visble_length = self.dataset_length
        if not self.dataset_length:
            raise ValueError('Empty ADIOS dataset')

    @staticmethod
    def _shape(info):
        return [int(part) for part in info.get('Shape', '').replace(',', ' ').split()]

    def __len__(self):
        return self.visble_length

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['reader'] = None
        return state

    def __getitem__(self, index):
        from adios2 import FileReader
        if self.reader is None:
            self.reader = FileReader(self.data_path)
        index %= self.dataset_length
        for field in self.fields:
            if index < field['length']:
                break
            index -= field['length']
        col = index % field['nw']; index //= field['nw']
        row = index % field['nh']; index //= field['nh']
        time = field['t0'] + (index % field['nt']) * self.n_frame
        section = field['s0'] + index // field['nt']
        if self.train_mode:
            top = np.random.randint(max(1, field['h']-self.size+1))
            left = np.random.randint(max(1, field['w']-self.size+1))
        else:
            top, left = row*self.size, col*self.size
        valid_t = min(self.n_frame, field['t1']-time)
        valid_h, valid_w = min(self.size, field['h']-top), min(self.size, field['w']-left)
        starts = dict(variable=field['variable'], section=section, time=time, height=top, width=left)
        counts = dict(variable=1, section=1, time=valid_t, height=valid_h, width=valid_w)
        start = [starts[axis] for axis in field['axes']]
        count = [counts[axis] for axis in field['axes']]
        if field['steps_time']:
            values = [self.reader.read(field['name'], start=start, count=count,
                                      step_selection=[field['step_start']+time+i, 1]) for i in range(valid_t)]
            value = canonicalize(np.stack(values), ['time', *field['axes']])[0, 0]
        else:
            value = canonicalize(np.asarray(self.reader.read(field['name'], start=start, count=count,
                                  step_selection=[field['step_start'], 1])), field['axes'])[0, 0]
        data = torch.from_numpy(np.asarray(value, dtype=np.float32).copy())
        data = F.pad(data[None, None], (0,self.size-valid_w,0,self.size-valid_h,0,self.n_frame-valid_t),
                     mode='replicate')[0,0]
        scale = data.max()-data.min()
        offset = data.mean()
        # Constant patches have a well-defined zero normalized representation.
        if scale == 0:
            scale = torch.ones_like(scale)
        result = {'input': ((data-offset)/scale)[None],
                  'offset': offset.reshape(1,1,1,1), 'scale': scale.reshape(1,1,1,1)}
        if not self.train_mode:
            result['valid_shape'] = torch.tensor([valid_t, valid_h, valid_w])
        return result
