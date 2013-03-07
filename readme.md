Identity Merging Algorithm
===

This Python tool merges aliases belonging to the same individual. Aliases are considered as `(name, emailAddress)` tuples. Based on several similarity measures, the algorithm returns for each combination of aliases a similarity value.

To compute the similarity between two aliases, the Levenshtein edit distance and Latent Semantic Analysis are used. Identity data sets are typically very large, creating matrices with large dimensions (100k+) which do not fit into memory. The algorithm has been optimised for speed and out-of-core computations.

### How does it work?

It starts off by walking through all aliases, grouping together all aliases having an equal email address. Each email address is considered as a *document*. The names and email address prefix are considered the *terms* of the document.

Having the documents, containing the corresponding terms, the data is transformed to [Vector Space Model](http://en.wikipedia.org/wiki/Vector_space_model) (VSM), creating a [document-term matrix](http://en.wikipedia.org/wiki/Document-term_matrix) which describes the frequency of terms that occur in the documents. In addition to term frequency, the similarity between terms is computed using the [Levenshtein distance](http://en.wikipedia.org/wiki/Levenshtein_distance), adding terms with a normalised Levenshtein similarity above a given threshold to the document by the similarity fraction. For example, consider the document `{(john, 1), (smith, 1)}`. A term, `johnny` would be included in the document: `{(john, 1), (smith, 1), (johnny, 0.66)}` as `LevenshteinDistance(john, johnny) = 0.66`.

To prevent redundant computations of the Levenshtein distance between two terms, a term-term matrix precomputes the Levenshtein distance between each combination of terms. Using the term-term matrix, the document-term matrix is augmented by adding for each term in the document the similar terms from the term-term matrix.

On the augmented document-term matrix, the [tf-idf](http://en.wikipedia.org/wiki/Tf%E2%80%93idf) model is applied, reducing the weight of terms that occur in a big number of documents. In other words, the weight of common words is reduced.

After applying the tf-idf model, [Latent Semantic Analysis](http://en.wikipedia.org/wiki/Latent_semantic_analysis) (LSA) is applied to the data. LSA basically computes the [Singular Value Decomposition](http://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) and applies a `k`-rank reduction. The rank reduction removes the least significant dimensions (terms), which is believed to remove noise from the data set.

Finally, the similarity between the documents is computed using the [cosine similarity](http://en.wikipedia.org/wiki/Cosine_similarity). This similarity is returned if it is above a given threshold.

### Dependencies

- `Python 2.7` http://www.python.org/getit/
- `gensim` https://pypi.python.org/pypi/gensim
	- `scipy` http://www.scipy.org/Download
	- `numpy` http://www.scipy.org/Download
- `unidecode` https://pypi.python.org/pypi/Unidecode/
- `python-Levenshtein` https://pypi.python.org/pypi/python-Levenshtein

### Licenses

- The Python tool is licensed under the [GNU Lesser General Public License](http://www.gnu.org/licenses/lgpl.txt) version 3