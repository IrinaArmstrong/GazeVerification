
from typeguard import typechecked
from gaze_verification.parsers.parser_abstract import ParserAbstract

@typechecked
class JuDo1000_parser(ParserAbstract):
    """
    Parser for csv-formatted datasets for classification (multi-class) problems (e. g. WASSA 2021).
    The first column contains text, the second one contains class label.

    Rows with empty label column will be skipped.

    Expected csv table:
        +-------------+----------------+
        | TEXT_COLUMN | TARGET_COLUMN  |
        +=============+================+
        | some text 1 | class 1        |
        +-------------+----------------+
        | some text 2 | class 3        |
        +-------------+----------------+
        | some text 3 | class 2        |
        +-------------+----------------+
        | some text 4 |                |
        +-------------+----------------+
        | some text 5 | class 1        |
        +-------------+----------------+

    We also support subtypes for classification, e. g. "Country" is a subclass of "LOC".
    If you wish to include them into the data, please, separate class labels ant their
    suptypes with symbol ":" e.g. "LOC:Country".

    Possible csv table:
        +-------------+-------------------+
        | TEXT_COLUMN | TARGET_COLUMN     |
        +=============+===================+
        | some text 1 | class 1:subtype a |
        +-------------+-------------------+
        | some text 2 | class 3           |
        +-------------+-------------------+
        | some text 3 | class 2           |
        +-------------+-------------------+
        | some text 4 |                   |
        +-------------+-------------------+
        | some text 5 | class 1:subtype b |
        +-------------+-------------------+

    :param tokenizer_name: Name of using tokenizer,
        defaults to WordPunct
        (see :class:`~auto_ner.tokenizers.word_tokenizers.WordTokenizers`)
    :type tokenizer_name: str

    :param text_column_name: name of column, containing texts for classification
    :type text_column_name: str

    :param target_column_name: name of column, containing target multiclass labels
    :type target_column_name: str

    :param csv_reading_kwargs: kwargs for read_csv pandas function
    :type csv_reading_kwargs: dictlike
    """
    INPUT_SEP_ATTR = ":"

    def __init__(
            self,
            *args,
            target_column_name: str = "target",
            csv_reading_kwargs: Dict[str, Any] = dict(),
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._target_column = target_column_name
        self._csv_reading_kwargs = csv_reading_kwargs