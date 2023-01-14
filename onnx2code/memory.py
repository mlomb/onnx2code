import math
from dataclasses import dataclass

# We implement different memory strategies used in TFLite
# Since we are using CPU we aim to the Memory Offset Calculation approach
#
# See:
#  * https://arxiv.org/pdf/2001.03288.pdf (main paper)
#  * https://blog.tensorflow.org/2020/10/optimizing-tensorflow-lite-runtime.html (blog post)
#  * https://github.com/tensorflow/tensorflow/blob/1b36c9fb27ce899e19ddf65da3c0920861210472/tensorflow/lite/delegates/gpu/common/memory_management (ref code)


@dataclass
class TensorUsageRecord:
    first_op: int
    last_op: int
    size: int
    index: int = -1  # used to store the original index after sorting


Records = list[TensorUsageRecord]
Offsets = list[int | None]
Result = tuple[int, Offsets]


##########################
# Naive
##########################
def naive(records: Records) -> Result:
    total_consumption = 0
    offsets: Offsets = [None] * len(records)

    for i, r in enumerate(records):
        offsets[i] = total_consumption
        total_consumption += r.size

    return total_consumption, offsets


##########################
# Greed by Size
#
# TFLite C impl: https://github.com/tensorflow/tensorflow/blob/1b36c9fb27ce899e19ddf65da3c0920861210472/tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.cc#L69
##########################
def greedy_by_size(records: Records) -> Result:
    # save original indexes
    for i, r in enumerate(records):
        r.index = i

    # sort records in decreasing order of size
    records.sort(key=lambda r: r.size, reverse=True)

    # result
    total_consumption = 0
    offsets: Offsets = [None] * len(records)

    # indexes already allocated, ordered by offset
    ordered_allocs: list[int] = []

    for t_i, t in enumerate(records):
        prev_offset = 0
        best_offset = None
        smallest_gap = math.inf

        for allocated_id in ordered_allocs:
            rec = records[allocated_id]

            if rec.last_op < t.first_op or rec.first_op > t.last_op:
                # no overlap, skip
                continue

            cur_offset = offsets[rec.index]
            assert cur_offset is not None

            if cur_offset >= prev_offset:
                gap = cur_offset - prev_offset

                if gap >= t.size and gap < smallest_gap:
                    smallest_gap = gap
                    best_offset = prev_offset

            prev_offset = max(prev_offset, cur_offset + rec.size)

        # if no suitable gap found, allocate at the end
        if best_offset is None:
            best_offset = prev_offset

        offsets[t.index] = best_offset
        total_consumption = max(total_consumption, best_offset + t.size)

        ordered_allocs.append(t_i)

        # sort by offset
        ordered_allocs.sort(key=lambda i: offsets[records[i].index])  # type: ignore

    return total_consumption, offsets


##########################
# Greed by Breadth
##########################
def greedy_by_breadth(records: Records) -> Result:
    raise NotImplementedError()


def find_best_layout(records: Records) -> Result:
    """
    Find the best memory layout using different strategies.
    """
    alternatives = [
        naive(records),
        greedy_by_size(records),
        # greedy_by_breadth(records),
    ]

    return min(alternatives, key=lambda r: r[0])


if __name__ == "__main__":
    test = [
        TensorUsageRecord(0, 1, 32),
        TensorUsageRecord(1, 4, 28),
        TensorUsageRecord(2, 5, 36),
        TensorUsageRecord(3, 5, 16),
        TensorUsageRecord(4, 5, 8),
        TensorUsageRecord(5, 7, 64),
        TensorUsageRecord(6, 8, 10),
        TensorUsageRecord(7, 8, 40),
    ]

    print(find_best_layout(test))
