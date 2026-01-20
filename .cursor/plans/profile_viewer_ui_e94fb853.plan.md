---
name: Profile Viewer UI
overview: Create a single-file HTML dashboard to visualize profiler results, showing metrics relevant to the assignment questions (CUDA kernel times, forward/backward comparison, operation breakdowns).
todos:
  - id: create-html
    content: Create scripts/profile_viewer.html with Tailwind CSS, Chart.js CDN, and base structure
    status: completed
  - id: summary-cards
    content: Implement Tailwind-styled summary cards (CUDA/CPU times, model metadata)
    status: completed
  - id: kernels-table
    content: Implement sortable/filterable table with Tailwind styling
    status: completed
  - id: breakdown-charts
    content: Add Chart.js doughnut chart for operation breakdown + horizontal bar for top kernels
    status: pending
  - id: comparison-chart
    content: Add grouped bar chart comparing forward vs backward pass by category
    status: pending
  - id: attention-chart
    content: Add chart comparing softmax vs matmul within attention (assignment Q.e)
    status: pending
---

# Profile Viewer UI

## Architecture

Single HTML file (`scripts/profile_viewer.html`) with:

- **Tailwind CSS** via CDN (Play CDN for zero build step)
- **Chart.js** via CDN for visualizations (pie, bar, doughnut charts)
- Inline JS for data loading and rendering
- Dark theme using Tailwind's dark mode utilities
- No build step - just open in browser

### CDN Dependencies

```html
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
```

## Key Features

### 1. File Upload / Data Loading

- Drag-and-drop JSON file upload
- Auto-detect array vs single result format

### 2. Summary Cards (Assignment Q.a)

- Total CUDA time (forward pass)
- Total CUDA time (backward pass)
- Total CPU time
- Number of events
- Model metadata (size, params, context_length)

### 3. Top Kernels Table (Assignment Q.b, Q.c)

- Sortable columns: name, cuda_time, count, percentage
- Search/filter by kernel name
- Highlight matrix multiply kernels (cutlass, gemm, bmm)
- Show invocation count per kernel

### 4. Operation Breakdown Charts (Assignment Q.c, Q.e)

**Doughnut Chart** - Overall operation category breakdown:

- Matrix multiply (cutlass, gemm, bmm, einsum)
- Attention (sdpa/*, attn/*)
- Normalization (rmsnorm)
- FFN (ffn/*, linear)
- Softmax (sdpa/softmax)
- Other

**Horizontal Bar Chart** - Top 15 kernels by CUDA time

**Attention Detail Chart** - Softmax vs MatMul within attention (Q.e)

### 5. Forward vs Backward Comparison (Assignment Q.d)

**Grouped Bar Chart** comparing forward vs backward:

- Side-by-side bars for each operation category
- Show absolute times and ratio (backward/forward)
- Highlight which operations grow most in backward pass

### 6. Multi-Model/Context Support

- Dropdown to select model size if multiple in JSON
- Compare across context lengths

## File Structure

```
scripts/
└── profile_viewer.html   # Single file, ~400-500 lines
```

## UI Layout

```
┌─────────────────────────────────────────────────────┐
│  [Drop JSON file here]  or  [Choose File]           │
├─────────────────────────────────────────────────────┤
│  Model: small │ Params: 128M │ Context: 128         │
├──────────────────────┬──────────────────────────────┤
│  Forward CUDA Time   │  Backward CUDA Time          │
│  695.4 ms            │  2,105.3 ms                  │
├──────────────────────┴──────────────────────────────┤
│  [Operation Breakdown Pie Chart]                    │
├─────────────────────────────────────────────────────┤
│  Top Kernels Table (sortable, filterable)           │
│  ┌────────────────┬────────┬───────┬───────┬──────┐ │
│  │ Name           │ CUDA   │ %     │ Count │ Avg  │ │
│  ├────────────────┼────────┼───────┼───────┼──────┤ │
│  │ block/attention│ 160ms  │ 23.0% │ 120   │ 1.3ms│ │
│  │ attn/rope      │ 91ms   │ 13.1% │ 120   │ 0.8ms│ │
│  │ ...            │        │       │       │      │ │
│  └────────────────┴────────┴───────┴───────┴──────┘ │
└─────────────────────────────────────────────────────┘
```

## Implementation Notes

- **Tailwind CSS** for all styling (dark theme: `bg-gray-900`, cards: `bg-gray-800`)
- **Tailwind Grid** for responsive dashboard layout (`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4`)
- **Chart.js** with custom dark theme colors
- Tailwind typography for clean tables (`divide-y divide-gray-700`)
- LocalStorage to remember last loaded file path
- Export summary as text for assignment answers
- Responsive design that works on different screen sizes