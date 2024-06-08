# Models theory

## Deformable Self-Attention

The Deformable Attention Module is an attention module introduced with the Deformable DETR architecture, which seeks to overcome one issue base Transformer attention has, that is, it looks over all possible spatial locations.

 The deformable attention module only attends to a small set of key sampling points around a reference point, regardless of the spatial size of the feature maps as shown in the figure below. By assigning only a small fixed number of keys for each query, the issues of convergence and feature spatial resolution can be mitigated.

![DeformAtt](.\.\image\Deformable-Attention.png)

Given an input feature map $x \in \mathbb{R}^{C\times H \times W}$, let $q$ index a query element with content feature $z_q$ and a 2-d reference point $p_q$, the deformable attention feature is calculated by:

$$ DeformAttn(z_q, p_q, x) = \sum^M_{m=1} W_m \Biggl[\sum^{K}_{k=1}A_{mqk}\cdot W'_mx(p_q + \Delta p_{mqk})\Biggr] $$

where $m$ indexes the attention head, $k$ indexes the sampled keys, and $K$ is the total sampled key number $K \ll HW$. $\Delta p_{mqk}$ and $A_{mqk}$ denote the sampling offset and attention weight of the $k^{th}$ sampling point in the $m^{th}$ attention head, respectively. The scalar attention weight $A_{mqk}$ lies in the range $[0,1]$, normalized by $\sum^K_{k=1} A_{mqk} = 1.$  $ \Delta p_{mqk} \in \mathbb{R}^2$
 are of 2-d real numbers with unconstrained range. As $p_q + \Delta p_{mqk} $
 is fractional, bilinear interpolation is applied in computing $x(p_q + \Delta p_{mqk}) $
. Both $\Delta p_{mqk}$ and $A_{mqk}$ are obtained via linear projection over the query feature $z_q$ In implementation, the query feature $z_q$ is fed to a linear projection operator of $3MK$ channels, where the first $2MK$ channels encode the sampling offsets $\Delta p_{mqk}$, and the remaining $MK$
 channels are fed to a softmax operator to obtain the attention weights $A_{mqk}$.