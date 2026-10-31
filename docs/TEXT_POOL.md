# The text pool

Every word on a generated page comes from a text pool: one JSON object with
six lists. The generator samples passages from them and never invents text, so
word lengths, diacritics and punctuation look like the source. Use text in the
register you want the pages to be in.

```json
{
  "paras":  ["a paragraph of body text …", "…"],
  "heads":  ["a title", "…"],
  "foots":  ["the text of a footnote …", "…"],
  "items":  ["a list item", "…"],
  "rows":   [["a title in an index", "١٢"], ["cell", "cell", "cell"], "…"],
  "verses": [["first half of a verse", "second half"], "…"]
}
```

| key | what it becomes | each entry |
|---|---|---|
| `paras` | body paragraphs, columns, glosses, captions | a string; longer is better |
| `heads` | headings, running heads, titles | a short string |
| `foots` | the footnote block | a string; the generator numbers it |
| `items` | list items, short lines, index entries | a string |
| `rows` | tables and indexes (*fahras*) | a list of cell strings |
| `verses` | poetry, set as two halves per line | `[first half, second half]` |

Every list needs at least 8 entries. A page uses only the passages its chosen
typeface can draw: a character the face lacks would otherwise render as an
empty box while the labels still claim the text. If fewer than 8 passages of a
list survive that filter, the page uses the whole list.

`tools/make_text_pool.py` builds a pool from plain UTF-8 text files, splitting
them into the six kinds by length and punctuation. It takes verse from a
separate file (`--verses`, one verse per line, its halves split by a tab or
` *** `). Without that file, pairs of short clauses stand in, and read as verse
only by their layout.
