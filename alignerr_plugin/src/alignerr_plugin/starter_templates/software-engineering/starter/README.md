# Slug normalizer

This repository exposes `normalizer.normalize_slug(value)`.

Run its public tests with:

```bash
python3 -m unittest discover -s tests -v
```

The seeded implementation passes the public smoke tests but mishandles parts of
the documented punctuation and empty-value contract.
