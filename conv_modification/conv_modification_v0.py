import nn
from torch import nn
from torch.nn import init


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_modification=True):
        super(CustomConv2d, self).__init__()
        self.padding = padding
        self.use_modification = use_modification
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        
        if use_modification:
            self.edge_filters = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias) for _ in range(padding)])

        self._initialize_weights()
        
    def _initialize_weights(self):
        init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0)
        if self.use_modification:
            for edge_filter in self.edge_filters:
                init.kaiming_normal_(edge_filter.weight, nonlinearity='leaky_relu')
                if edge_filter.bias is not None:
                    init.constant_(edge_filter.bias, 0)

    def forward(self, x):
        conv_out = self.conv(x)

        if self.padding > 0 and self.use_modification:
            edge_filter_sum = torch.zeros_like(conv_out)
            _, _, h, w = x.size()
            for i in range(1, self.padding + 1):
                edge_filter = self.edge_filters[i - 1](x)
                
                # Top edge (excluding corners)
                edge_filter_sum[:, :, i-1:i, i:-i] += edge_filter[:, :, i-1:i, i:-i]
                # Bottom edge (excluding corners)
                edge_filter_sum[:, :, -i:, i:-i] += edge_filter[:, :, -i:, i:-i]
                # Left edge (including corners)
                edge_filter_sum[:, :, :, i-1:i] += edge_filter[:, :, :, i-1:i]
                # Right edge (including corners)
                edge_filter_sum[:, :, :, -i:] += edge_filter[:, :, :, -i:]

                return conv_out + edge_filter_sum

        return conv_out
