# first line: 105
    def _compute():
        ids, attn = tokenize_to_windows(text, max_len=max_len, stride=stride)
        V = encode_windows_batched(ids, attn, batch_size=batch_size)
        agg = aggregate_windows(V, topk=3)
        return agg
