# lib/builder.py 完整代码（纯PyTorch实现，无mmcv依赖）
class Registry:
    """纯PyTorch实现的注册器，替代mmcv.utils.Registry"""

    def __init__(self, name):
        self.name = name  # 注册器名称（如'backbone'）
        self._module_dict = dict()  # 存储注册的模块：{模块名: 模块类}

    def register_module(self, cls=None):
        """
        装饰器，用于注册模块
        用法1：@BACKBONES.register_module()
        用法2：BACKBONES.register_module(module_class)
        """

        def _register(cls):
            # 检查模块名是否重复
            if cls.__name__ in self._module_dict:
                raise KeyError(f'{cls.__name__} is already registered in {self.name}')
            self._module_dict[cls.__name__] = cls
            return cls

        if cls is None:
            return _register  # 装饰器模式
        else:
            return _register(cls)  # 直接注册模式

    def build(self, cfg):
        """
        根据配置字典构建模块
        Args:
            cfg: dict，必须包含 'type' 键（指定模块名），其余为模块初始化参数
        Returns:
            实例化的模块对象
        """
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        cfg_ = cfg.copy()
        module_name = cfg_.pop('type')  # 取出模块名

        # 检查模块是否已注册
        if module_name not in self._module_dict:
            raise KeyError(f'{module_name} is not registered in {self.name}')

        # 实例化模块
        module_cls = self._module_dict[module_name]
        module = module_cls(**cfg_)
        return module


# 定义BGSNet需要的注册器（匹配swins_extractor.py的导入）
BACKBONES = Registry('backbone')
HEADS = Registry('head')
NECKS = Registry('neck')
LOSSES = Registry('loss')
SEGMENTORS = Registry('segmentor')


# 以下为可选的辅助构建函数（匹配原builder.py的接口）
def build_backbone(cfg):
    """构建骨干网络"""
    return BACKBONES.build(cfg)


def build_head(cfg):
    """构建头部网络"""
    return HEADS.build(cfg)


def build_neck(cfg):
    """构建颈部网络"""
    return NECKS.build(cfg)


def build_loss(cfg):
    """构建损失函数"""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """构建分割器（兼容原接口）"""
    import warnings
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))