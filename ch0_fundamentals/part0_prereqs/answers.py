# [0.0] Prerequisites - https://learn.arena.education/chapter0_fundamentals/00_prereqs/
#
# Paste the page's setup code into the first cell, then work through the exercises cell by cell.
# Run a cell with the "Run Cell" link above each `# %%`, or select lines and press Shift+Enter.

# %%import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter0_fundamentals"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import einops
except:
    %pip install einops jaxtyping

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/ARENA-education/ARENA_materials/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/ARENA_materials-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/ARENA_materials-{branch}


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")

# %%
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")
# %%
print(arr[0].shape)
display_array_as_img(arr[0])  # plotting the first image in the batch
# %%
print(arr[0, 0].shape)
display_array_as_img(arr[0, 0])  # plotting the first channel of the first image, as monochrome
# %%
arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
print(arr_stacked.shape)
display_array_as_img(arr_stacked)  # plotting all images, stacked in a row

# %%
#(1) Column-stacking
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
display_array_as_img(arr1)
# %%
#(2) Column-stacking and copying
arr2 = einops.repeat(arr[0], "c h w -> c (repeat h) w", repeat=2)
display_array_as_img(arr2)
# %%
#(3) Raw-stacking and double-copying
arr3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)
# %%
#(4) Streching
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
display_array_as_img(arr4)
# %%
#(5) Split channels
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
display_array_as_img(arr5)
# %%
#(6) Stack into rows & cols
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
display_array_as_img(arr6)
# %%
#(7) Transpose
arr7 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr7)
# %%
#(8) Shrinking
arr8 = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", b1=2, h2=2, w2=2)
display_array_as_img(arr8)
# %%
#(A1) rearrange
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    T = t.arange(3, 9)
    return einops.rearrange(T, "(h w) -> h w", h=3, w=2)


tests.test_rearrange_1(rearrange_1)
# %%
#(A2) rearrange
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    return einops.rearrange(t.arange(1,7), "(h w) -> h w", h=2, w=3)


tests.test_rearrange_2(rearrange_2)
# %%
#(B1) temperature average
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return einops.reduce(temps, "(h w) -> h", "mean", h=int(len(temps)/7), w=7)


tests.test_temperatures_average(temperatures_average)
# %%
#(B2) temperature differance
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    avg_temps = einops.reduce(temps, "(h 7) -> h", "mean")
    broad_avg_temps = einops.repeat(avg_temps, "h -> (h 7)")
    return temps - broad_avg_temps


tests.test_temperatures_differences(temperatures_differences)
# %%
#(B3) temperature normalized
def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    avg = einops.reduce(temps, "(h 7) -> h", "mean")
    std = einops.reduce(temps, "(h 7) -> h", t.std)
    b_avg = einops.repeat(avg, "h -> (h 7)")
    b_std = einops.repeat(std, "h -> (h 7)")
    return (temps - b_avg)/b_std


tests.test_temperatures_normalized(temperatures_normalized)
# %%
#(C1) normalize a matrix
def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    return matrix/matrix.norm(dim=1 , keepdim=True)


tests.test_normalize_rows(normalize_rows)
# %%
#(C2) pairwise cosine similarity
def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.

    matrix: shape (m, n)
    """
    norm_rows = matrix/matrix.norm(dim=1 , keepdim=True)
    return norm_rows @ norm_rows.T


tests.test_cos_sim_matrix(cos_sim_matrix)
# %%
#(D) sample distribution
def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use t.rand and t.cumsum to do this without any explicit loops.
    """
    cum = t.cumsum(probs, dim=0)
    sample = t.rand(n, 1)
    return (sample > cum).sum(dim=-1)


tests.test_sample_distribution(sample_distribution)
# %%
#(E) classifier accuracy
def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use t.argmax.
    """
    return (scores.argmax(dim=1) == true_classes).sum() / len(true_classes)


tests.test_classifier_accuracy(classifier_accuracy)
# %%
#(F1) total price indexing
def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    return t.gather(prices, 0, items).sum().item()


tests.test_total_price_indexing(total_price_indexing)
# %%
#(F2) gather 2D
def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    """Perform a gather operation along the second dimension.

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). out[i][j] = matrix[i][indexes[i][j]]

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output. This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    """
    # YOUR CODE HERE - add assert statement(s) here for `indices` and `matrix`
    assert indexes.shape[0] <= matrix.shape[0]
    assert indexes.ndim == matrix.ndim
    out = matrix.gather(1, indexes)
    # YOUR CODE HERE - add assert statement(s) here for `out`
    assert out.shape == indexes.shape
    return out


tests.test_gather_2d(gather_2d)
# %%
#(F3) total price gather
def total_price_gather(prices: Tensor, items: Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather."""
    assert items.max() < prices.shape[0]
    return t.gather(prices, 0, items).sum().item()


tests.test_total_price_gather(total_price_gather)
# %%
#(G) indexing
def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )
    """
    return matrix[tuple(coords.T)]


tests.test_integer_array_indexing(integer_array_indexing)
# %%
#(H1) batched logsumexp
def batched_logsumexp(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    m = matrix.max(dim=-1).values
    exp = t.exp(matrix - einops.rearrange(m, "b -> b 1"))
    return t.log(t.sum(exp, dim=-1)) + m

tests.test_batched_logsumexp(batched_logsumexp)
# %%
#(H2) batched softmax
def batched_softmax(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute softmax(row).

    Do this without using PyTorch's softmax function.
    Instead, use the definition of softmax: https://en.wikipedia.org/wiki/Softmax_function

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.
    """
    exp = t.exp(matrix)
    sums = t.sum(exp, dim=-1, keepdim=True)
    return exp / sums


tests.test_batched_softmax(batched_softmax)
# %%
#(H3) batched logsoftmax
def batched_logsoftmax(matrix: Tensor) -> Tensor:
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n).

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    """
    m = matrix.max(dim=-1).values
    exp = t.exp(matrix - einops.rearrange(m, "b -> b 1"))
    sums = t.sum(exp, dim=-1, keepdim=True)
    return matrix - t.log(sums) - einops.rearrange(m, "b -> b 1")


tests.test_batched_logsoftmax(batched_logsoftmax)
# %%
#(H4) batched cross entropy loss
def batched_cross_entropy_loss(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Compute the cross entropy loss for each example in the batch.

    logits: shape (batch, classes). logits[i][j] is the unnormalized prediction for example i and class j.
    true_labels: shape (batch, ). true_labels[i] is an integer index representing the true class for example i.

    Return: shape (batch, ). out[i] is the loss for example i.

    Hint: convert the logits to log-probabilities using your batched_logsoftmax from above.
    Then the loss for an example is just the negative of the log-probability that the model assigned to the true class. Use torch.gather to perform the indexing.
    """
    assert logits.shape[0] == true_labels.shape[0]
    assert true_labels.max() < logits.shape[1]
    log_probs = batched_logsoftmax(logits)
    indices = einops.rearrange(true_labels, "b -> b 1")
    return -einops.rearrange(t.gather(log_probs, -1, indices), "b 1 -> b")

tests.test_batched_cross_entropy_loss(batched_cross_entropy_loss)
# %%
#(I1) collect rows
def collect_rows(matrix: Tensor, row_indexes: Tensor) -> Tensor:
    """Return a 2D matrix whose rows are taken from the input matrix in order according to row_indexes.

    matrix: shape (m, n)
    row_indexes: shape (k,). Each value is an integer in [0..m).

    Return: shape (k, n). out[i] is matrix[row_indexes[i]].
    """
    return matrix[row_indexes]


tests.test_collect_rows(collect_rows)
# %%
#(I2) collect columns
def collect_columns(matrix: Tensor, column_indexes: Tensor) -> Tensor:
    """Return a 2D matrix whose columns are taken from the input matrix in order according to column_indexes.

    matrix: shape (m, n)
    column_indexes: shape (k,). Each value is an integer in [0..n).

    Return: shape (m, k). out[:, i] is matrix[:, column_indexes[i]].
    """
    assert column_indexes.max() < matrix.shape[1]
    return matrix[:, column_indexes]


tests.test_collect_columns(collect_columns)
# %%
#einsum
def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    return einops.einsum(mat, "i i ->")


def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return einops.einsum(mat, vec, "i j, j -> i")


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return einops.einsum(mat1, mat2, "i j, j k -> i k")


def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.inner`.
    """
    return einops.einsum(vec1, vec2, "i, i ->")


def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.outer`.
    """
    return einops.einsum(vec1, vec2, "i, j -> i j")


tests.test_einsum_trace(einsum_trace)
tests.test_einsum_mv(einsum_mv)
tests.test_einsum_mm(einsum_mm)
tests.test_einsum_inner(einsum_inner)
tests.test_einsum_outer(einsum_outer)
# %%
