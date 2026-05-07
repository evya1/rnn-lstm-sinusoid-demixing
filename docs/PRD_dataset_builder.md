# PRD — Dataset Builder

## Purpose

Define requirements for constructing the sliding-window dataset used to
train and evaluate all models.

## Scope

Covers module: `src/rnn_lstm_sinusoid_demixing/data/dataset_builder.py`.

## Dataset Structure

Each training example is a triple `(input_window, selector, target_window)`:

| Array | Shape | Description |
|-------|-------|-------------|
| `inputs` | `(num_examples, window_size)` | Sliding windows over the noisy composite signal |
| `selectors` | `(num_examples, num_components)` | One-hot vector identifying the target component |
| `targets` | `(num_examples, window_size)` | Corresponding window from the selected clean component |

`num_examples = num_windows × num_components`  
`num_windows = num_samples − window_size + 1`

## Model-specific Input Formatting

The dataset builder returns raw arrays. Model code combines them as needed:

| Model | Input construction |
|-------|--------------------|
| FC | `np.concatenate([input_window, selector])` → shape `(window_size + num_components,)` |
| RNN/LSTM | interleave: each timestep `[sigma_t, c_1, c_2, c_3, c_4]` → shape `(window_size, 1 + num_components)` |

## Function Interface

### `make_one_hot(index, num_classes) -> NDArray`

Returns a 1-D float32 vector of length `num_classes` with `vec[index] = 1.0`.
Raises `ValueError` if `index` is out of range.

### `extract_windows(signal, window_size) -> NDArray`

Slides a fixed window over a 1-D signal using `sliding_window_view` (zero-copy).
Returns shape `(num_samples − window_size + 1, window_size)` as float32.
Raises `ValueError` if `window_size > len(signal)`.

### `build_dataset(composite_signal, clean_components, window_size, num_components) -> tuple`

Assembles the full dataset by:
1. Extracting all sliding windows from `composite_signal` → `inputs`.
2. Tiling those windows once per component → shape `(num_examples, window_size)`.
3. Building one-hot `selectors` via `np.eye` repeated per window.
4. Extracting windows from each clean component → `targets`.

Returns `(inputs, selectors, targets)`, all float32, no NaN or Inf.

## Output Contracts

- All arrays are float32.
- No NaN or Inf in any output.
- Each row of `selectors` is a valid one-hot vector (sum = 1, values in {0, 1}).
- `targets[k]` is the window from `clean_components[j]` at the same position as
  `inputs[k]`, where `j` is the argmax of `selectors[k]`.

## Acceptance Tests

- `inputs` shape is `(num_windows * num_components, window_size)`.
- `selectors` shape is `(num_windows * num_components, num_components)`.
- `targets` shape is `(num_windows * num_components, window_size)`.
- Each selector row sums to 1 and contains only 0s and 1s.
- Target window matches the corresponding clean component window.
- No NaN or Inf values.
