import tensorflow as tf
import tensorflow_addons as tfa

def window_partition(x, window_size):
    B, H, W, C = tf.unstack(tf.shape(x), num=4)
    x = tf.reshape(x, shape=[-1, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x


class SE(tf.keras.layers.Layer):
    def __init__(self, oup=None, expansion=0.25, **kwargs):
        super().__init__(**kwargs)
        self.expansion = expansion
        self.oup = oup

    def build(self, input_shape):
        inp = input_shape[-1]
        self.oup = self.oup or inp
        self.avg_pool = tfa.layers.AdaptiveAveragePooling2D(1, name="avg_pool")
        self.fc = [
            tf.keras.layers.Dense(int(inp * self.expansion), use_bias=False, name='fc/0'),
            tf.keras.layers.Activation('gelu', name='fc/1'),
            tf.keras.layers.Dense(self.oup, use_bias=False, name='fc/2'),
            tf.keras.layers.Activation('sigmoid', name='fc/3')
            ]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        b, _, _, c = tf.unstack(tf.shape(inputs), num=4)
        x = tf.reshape(self.avg_pool(inputs), (b, c))
        for layer in self.fc:
            x = layer(x)
        x = tf.reshape(x, (b, 1, 1, c))
        return x*inputs
    
class ReduceSize(tf.keras.layers.Layer):
    def __init__(self, keep_dim=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_dim = keep_dim

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        dim_out = embed_dim if self.keep_dim else 2*embed_dim
        self.pad1 = tf.keras.layers.ZeroPadding2D(1, name='pad1')
        self.pad2 = tf.keras.layers.ZeroPadding2D(1, name='pad2')
        self.conv = [
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='valid', use_bias=False, name='conv/0'),
            tf.keras.layers.Activation('gelu', name='conv/1'),
            SE(name='conv/2'),
            tf.keras.layers.Conv2D(embed_dim, kernel_size=1, strides=1, padding='valid', use_bias=False, name='conv/3')
        ]
        self.reduction = tf.keras.layers.Conv2D(dim_out, kernel_size=3, strides=2, padding='valid', use_bias=False,
                                                name='reduction')
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm1')  # eps like PyTorch
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm2')
        super().build(input_shape)
    
    def call(self, inputs, **kwargs):
        x = self.norm1(inputs)
        xr = self.pad1(x)
        for layer in self.conv:
            xr = layer(xr)
        x = x + xr
        x = self.pad2(x)
        x = self.reduction(x)
        x = self.norm2(x)
        return x
    
class Mlp(tf.keras.layers.Layer):
    def __init__(self, hidden_features=None, out_features=None, act_layer='gelu', dropout=0., **kwargs):
        super().__init__(**kwargs)
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.dropout = dropout

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.hidden_features = self.hidden_features or self.in_features
        self.out_features = self.out_features or self.in_features
        self.fc1 = tf.keras.layers.Dense(self.hidden_features, name="fc1")
        self.act = tf.keras.layers.Activation(self.act_layer, name="act")
        self.fc2 = tf.keras.layers.Dense(self.out_features, name="fc2")
        self.drop1 = tf.keras.layers.Dropout(self.dropout, name="drop1")
        self.drop2 = tf.keras.layers.Dropout(self.dropout, name="drop2")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    
class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=0., scale_by_keep=True, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, x, training=None):
        if self.drop_prob==0. or not training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
        random_tensor = tf.floor(random_tensor)
        if keep_prob > 0.0 and self.scale_by_keep:
            x = (x / keep_prob) 
        return x * random_tensor
    
class Identity(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.identity(x)
    
    
class GCViT(tf.keras.Model):
    def __init__(self, window_size, dim, depths, num_heads,
        drop_rate=0., mlp_ratio=3., qkv_bias=True, qk_scale=None, attn_drop=0., path_drop=0.1, layer_scale=None,
        num_classes=1000, head_act='softmax', **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.layer_scale = layer_scale
        self.resize_query = resize_query
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.head_act = head_act

        self.patch_embed = PatchEmbed(embed_dim=embed_dim, name='patch_embed')
        self.pos_drop = tf.keras.layers.Dropout(drop_rate, name='pos_drop')
        path_drops = np.linspace(0., path_drop, sum(depths))
        keep_dims = [(False, False, False),(False, False),(True,),(True,),]
        self.levels = []
        for i in range(len(depths)):
            path_drop = path_drops[sum(depths[:i]):sum(depths[:i + 1])].tolist()
            level = Level(depth=depths[i], num_heads=num_heads[i], window_size=window_size[i], keep_dims=keep_dims[i],
                    downsample=(i < len(depths) - 1), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    drop=drop_rate, attn_drop=attn_drop, path_drop=path_drop, layer_scale=layer_scale,
                    name=f'levels/{i}')
            self.levels.append(level)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm')
        self.pool = tf.keras.layers.GlobalAveragePooling2D(name='pool')
        self.head = tf.keras.layers.Dense(num_classes, name='head', activation=head_act)

    def call(self, inputs, **kwargs):
        x = self.patch_embed(inputs)  # shape: (B, H, W, C)
        x = self.pos_drop(x)
        x = tf.cast(x, dtype=tf.float32)
        for level in self.levels:
            x = level(x)  # shape: (B, H_, W_, C_)
        x = self.norm(x)
        x = self.pool(x)  # shape: (B, C__)
        x = self.head(x)
        return x

    def build_graph(self, input_shape=(224, 224, 3)):
        """https://www.kaggle.com/code/ipythonx/tf-hybrid-efficientnet-swin-transformer-gradcam"""
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.name)

    def summary(self, input_shape=(224, 224, 3)):
        return self.build_graph(input_shape).summary()
    
    
class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pad = tf.keras.layers.ZeroPadding2D(1, name='pad')
        self.proj = tf.keras.layers.Conv2D(self.embed_dim, kernel_size=3, strides=2, name='proj')
        self.conv_down = ReduceSize(keep_dim=True, name='conv_down')
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.pad(inputs)
        x = self.proj(x)
        x = self.conv_down(x)
        return x
    
class Level(tf.keras.layers.Layer):
    def __init__(self, depth, num_heads, window_size, keep_dims, downsample=True, mlp_ratio=4., qkv_bias=True, 
                qk_scale=None, drop=0., attn_drop=0., path_drop=0., layer_scale=None, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.keep_dims = keep_dims
        self.downsample = downsample
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.layer_scale = layer_scale

    def build(self, input_shape):
        path_drop = [self.path_drop] * self.depth if not isinstance(self.path_drop, list) else self.path_drop
        self.blocks = [
            Block(window_size=self.window_size,
                      num_heads=self.num_heads,
                      global_query=bool(i % 2),
                      mlp_ratio=self.mlp_ratio, 
                      qkv_bias=self.qkv_bias, 
                      qk_scale=self.qk_scale, 
                      drop=self.drop,
                      attn_drop=self.attn_drop, 
                      path_drop=path_drop[i],
                      layer_scale=self.layer_scale, 
                      name=f'blocks/{i}')
            for i in range(self.depth)]
        self.down = ReduceSize(keep_dim=False, name='downsample')
        self.q_global_gen = GlobalQueryGen(self.keep_dims, name="q_global_gen")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        H, W = tf.unstack(tf.shape(inputs)[1:3], num=2)
        x = inputs
        q_global = self.q_global_gen(x)  # shape: (B, win_size, win_size, C)
        for i, blk in enumerate(self.blocks):
            if i % 2:
                x = blk([x, q_global])  # shape: (B, H, W, C)
            else:
                x = blk([x])  # shape: (B, H, W, C)
        if self.downsample:
            x = self.down(x)  # shape: (B, H//2, W//2, 2*C)
        return x
    
    
class FeatExtract(tf.keras.layers.Layer):
    def __init__(self, keep_dim=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_dim = keep_dim

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.pad1 = tf.keras.layers.ZeroPadding2D(1, name='pad1')
        self.pad2 = tf.keras.layers.ZeroPadding2D(1, name='pad2')
        self.conv = [
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='valid', use_bias=False, name='conv/0'),
            tf.keras.layers.Activation('gelu', name='conv/1'),
            SE(name='conv/2'),
            tf.keras.layers.Conv2D(embed_dim, kernel_size=1, strides=1, padding='valid', use_bias=False, name='conv/3')
        ]
        if not self.keep_dim:
            self.pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', name='pool')
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        xr = self.pad1(x)
        for layer in self.conv:
            xr = layer(xr)
        x = x + xr # if pad had weights it would've thrown error with .save_weights()
        if not self.keep_dim:
            x = self.pad2(x)
            x = self.pool(x)
        return x
    
class  GlobalQueryGen(tf.keras.layers.Layer):
    def __init__(self, keep_dims=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_dims = keep_dims
        
    def build(self, input_shape):
        self.to_q_global = [FeatExtract(keep_dim, name=f'to_q_global/{i}') \
                            for i, keep_dim in enumerate(self.keep_dims)]
        super().build(input_shape)
        
    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.to_q_global:
            x = layer(x)
        return x
    
class Block(tf.keras.layers.Layer):
    def __init__(self, window_size, num_heads, global_query, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., path_drop=0., act_layer='gelu', layer_scale=None, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.num_heads = num_heads
        self.global_query = global_query
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.act_layer = act_layer
        self.layer_scale = layer_scale

    def build(self, input_shape):
        B, H, W, C = input_shape[0]
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm1')
        self.attn = WindowAttention(window_size=self.window_size, 
                                   num_heads=self.num_heads,
                                   global_query=self.global_query,
                                   qkv_bias=self.qkv_bias, 
                                   qk_scale=self.qk_scale, 
                                   attn_dropout=self.attn_drop, 
                                   proj_dropout=self.drop,
                                   name='attn')
        self.drop_path1 = DropPath(self.path_drop)
        self.drop_path2 = DropPath(self.path_drop)
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm2')
        self.mlp = Mlp(hidden_features=int(C * self.mlp_ratio), dropout=self.drop, act_layer=self.act_layer, name='mlp')
        if self.layer_scale is not None:
            self.gamma1 = self.add_weight(
                'gamma1',
                shape=[C],
                initializer=tf.keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype)
            self.gamma2 = self.add_weight(
                'gamma2',
                shape=[C],
                initializer=tf.keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        self.num_windows = int(H // self.window_size) * int(W // self.window_size)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.global_query:
            inputs, q_global = inputs
        else:
            inputs = inputs[0]
        B, H, W, C = tf.unstack(tf.shape(inputs), num=4)
        x = self.norm1(inputs)
        # create windows and concat them in batch axis
        x = window_partition(x, self.window_size)  # (B_, win_h, win_w, C)
        # flatten patch
        x = tf.reshape(x, shape=[-1, self.window_size * self.window_size, C])  # (B_, N, C) => (batch*num_win, num_token, feature)
        # attention
        if self.global_query:
            x = self.attn([x, q_global])
        else:
            x = self.attn([x])
        # reverse window partition
        x = window_reverse(x, self.window_size, H, W, C)
        # FFN
        x = inputs + self.drop_path1(x * self.gamma1)
        x = x + self.drop_path2(self.gamma2 * self.mlp(self.norm2(x)))
        return x
    
    
class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, window_size, num_heads, global_query, qkv_bias=True, qk_scale=None, attn_dropout=0., proj_dropout=0.,
                 **kwargs):
        super().__init__(**kwargs)
        window_size = (window_size,window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        self.global_query = global_query
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout
    
    def build(self, input_shape):
        embed_dim = input_shape[0][-1]
        head_dim = embed_dim // self.num_heads
        self.scale = self.qk_scale or head_dim ** -0.5
        self.qkv_size = 3 - int(self.global_query)
        self.qkv = tf.keras.layers.Dense(embed_dim * self.qkv_size, use_bias=self.qkv_bias, name='qkv')
        self.relative_position_bias_table = self.add_weight(
            'relative_position_bias_table',
            shape=[(2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype)
        self.attn_drop = tf.keras.layers.Dropout(self.attn_dropout, name='attn_drop')
        self.proj = tf.keras.layers.Dense(embed_dim, name='proj')
        self.proj_drop = tf.keras.layers.Dropout(self.proj_dropout, name='proj_drop')
        self.softmax = tf.keras.layers.Activation('softmax', name='softmax')
        self.relative_position_index = self.get_relative_position_index()
        super().build(input_shape)
        
    def get_relative_position_index(self):
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'), axis=0)
        coords_flatten = tf.reshape(coords, [2, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        relative_coords_xx = (relative_coords[:, :, 0] + self.window_size[0] - 1)
        relative_coords_yy = (relative_coords[:, :, 1] + self.window_size[1] - 1) 
        relative_coords_xx = relative_coords_xx * (2 * self.window_size[1] - 1)
        relative_position_index = (relative_coords_xx + relative_coords_yy)
        return relative_position_index

    def call(self, inputs, **kwargs):
        if self.global_query:
            inputs, q_global = inputs
            B = tf.shape(q_global)[0] # B, N, C
        else:
            inputs = inputs[0]
        B_, N, C = tf.unstack(tf.shape(inputs), num=3) # B*num_window, num_tokens, channels
        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv, [B_, N, self.qkv_size, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        if self.global_query:
            k, v = tf.unstack(qkv, num=2, axis=0)  # for unknown shame num=None will throw error
            q_global = tf.repeat(q_global, repeats=B_//B, axis=0) # num_windows = B_//B => q_global same for all windows in a img
            q = tf.reshape(q_global, shape=[B_, N, self.num_heads, C // self.num_heads])
            q = tf.transpose(q, perm=[0, 2, 1, 3])
        else:
            q, k, v = tf.unstack(qkv, num=3, axis=0)
        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias,
                                            shape=[self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        attn = attn + relative_position_bias[tf.newaxis,]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3]) # B_, num_tokens, num_heads, channels_per_head
        x = tf.reshape(x, shape=[B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x