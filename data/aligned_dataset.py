import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        # Use the shortest list so A/B (and optional inst/feat) stay aligned
        self.dataset_size = len(self.A_paths)
        if hasattr(self, 'B_paths') and self.B_paths is not None:
            if len(self.B_paths) != len(self.A_paths):
                print('Warning: A_paths (%d) and B_paths (%d) have different lengths. Using min length.'
                      % (len(self.A_paths), len(self.B_paths)))
            self.dataset_size = min(self.dataset_size, len(self.B_paths))
        if hasattr(self, 'inst_paths') and self.inst_paths is not None:
            if len(self.inst_paths) != self.dataset_size:
                print('Warning: inst_paths length (%d) does not match dataset size (%d). Using min.'
                      % (len(self.inst_paths), self.dataset_size))
            self.dataset_size = min(self.dataset_size, len(self.inst_paths))
        if hasattr(self, 'feat_paths') and self.feat_paths is not None:
            if len(self.feat_paths) != self.dataset_size:
                print('Warning: feat_paths length (%d) does not match dataset size (%d). Using min.'
                      % (len(self.feat_paths), self.dataset_size))
            self.dataset_size = min(self.dataset_size, len(self.feat_paths))
        print('AlignedDataset size = %d (A=%d, B=%d)'
              % (self.dataset_size, len(self.A_paths),
                 len(self.B_paths) if hasattr(self, 'B_paths') and self.B_paths is not None else 0))

    def __getitem__(self, index):
        # Guard against out-of-range indices from DataLoader / resume
        index = index % self.dataset_size

        ### input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)

        ### if using instance maps
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'