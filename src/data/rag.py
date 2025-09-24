import uuid
from typing import Literal

import lancedb
from lancedb.table import Table
from lancedb.query import LanceQueryBuilder
from numpy import ndarray
from torch import Tensor


class RAGDatabase:
    def __init__(self, db_path: str, table_name: str, device: Literal['cpu', 'cuda'] = 'cpu'):
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
        self.table.embedding_functions['text_embedding'].function.device = device

    @staticmethod
    def format_result(result: LanceQueryBuilder, format: Literal["pandas", "pyarrow", "dict", "list"] = 'dict'):
        """
        Format the result of a search.
        :param result: The result of a search.
        :param format: The format of the result.
        :return: The formatted result.
        """
        if format == 'pandas':
            return result.to_pandas()
        elif format == 'pyarrow':
            return result.to_arrow()
        elif format == 'dict':
            return result.to_pandas().to_dict('records')
        elif format == 'list':
            return result.to_list()
        else:
            raise ValueError(f'Invalid format: {format}')

    def vector_search(self, vector: ndarray | str, vector_column_name: str = None, top_k: int = 10, table: Table = None,
                      where: str = None, select: list[str] = None, nprobes: int = 50, refine_factor: int = 30,
                      output_format: Literal["pandas", "pyarrow", "dict"] = 'dict'):
        """
        Search for a vector in the database.
        :param vector: The vector to search for.
        :param vector_column_name: The name of the vector column.
        :param top_k: The number of results to return.
        :param table: The table to search in.
        :param where: A where clause to filter the results.
        :param select: A list of columns to return.
        :param nprobes: The number of probes determines the distribution of vector space.
        :param refine_factor: Refine the results by reading extra elements and re-ranking them in memory.
        :param output_format: The format of the results.
        :return: A list of dictionaries containing the results.
        """
        table = self.table if table is None else table

        results = table.search(vector, vector_column_name).limit(top_k).nprobes(nprobes).refine_factor(refine_factor)

        if where is not None:
            results = results.where(where)
        if select is not None:
            results = results.select(select)

        return self.format_result(results, output_format)

    def text_search(self, text: str | Tensor | ndarray, top_k: int = 10, table: Table = None, where: str = None,
                    select: list[str] = None, nprobes: int = 50, refine_factor: int = 30,
                    output_format: Literal["pandas", "pyarrow", "dict"] = 'dict'):
        """
        Search for text in the database.
        :param text: The text or embedding to search for.
        :param top_k: The number of results to return.
        :param table: The table to search in.
        :param where: A where clause to filter the results.
        :param select: A list of columns to return.
        :param nprobes: The number of probes determines the distribution of vector space.
        :param refine_factor: Refine the results by reading extra elements and re-ranking them in memory.
        :param output_format: The format of the results.
        :return: A list of dictionaries containing the results.
        """
        return self.vector_search(text, vector_column_name="text_embedding",
                                  top_k=top_k, table=table, where=where, select=select, nprobes=nprobes,
                                  refine_factor=refine_factor, output_format=output_format)

    def image_search(self, image_embedding: ndarray, top_k: int = 10, table: Table = None, where: str = None,
                     select: list[str] = None, nprobes: int = 50, refine_factor: int = 30,
                     output_format: Literal["pandas", "pyarrow", "dict"] = 'dict'):
        """
        Search for an image in the database.
        :param image_embedding: The embedding of the image to search for.
        :param top_k: The number of results to return.
        :param table: The table to search in.
        :param where: A where clause to filter the results.
        :param select: A list of columns to return.
        :param nprobes: The number of probes determines the distribution of vector space.
        :param refine_factor: Refine the results by reading extra elements and re-ranking them in memory.
        :param output_format: The format of the results.
        :return: A list of dictionaries containing the results.
        """
        return self.vector_search(image_embedding, vector_column_name="image_embedding",
                                  top_k=top_k, table=table, where=where, select=select, nprobes=nprobes,
                                  refine_factor=refine_factor, output_format=output_format)

    def text_image_search(self, text: str | Tensor | ndarray, image_embedding: ndarray,
                          top_k: tuple[int, int] = (20, 10), table: Table = None, where: str = None,
                          select: list[str] = None, nprobes: int = 50, refine_factor: int = 30,
                          output_format: Literal["pandas", "pyarrow", "dict"] = 'dict'):
        """
        Search for a text and an image in the database.
        :param text: The text to search for.
        :param image_embedding: The embedding of the image to search for.
        :param top_k: The number of results to return.
        :param table: The table to search in.
        :param where: A where clause to filter the results.
        :param select: A list of columns to return.
        :param nprobes: The number of probes determines the distribution of vector space.
        :param refine_factor: Refine the results by reading extra elements and re-ranking them in memory.
        :param output_format: The format of the results.
        :return: A list of dictionaries containing the results.
        """
        pre_select = [*select, 'image_embedding', 'text_embedding', 'text'] if select is not None else None

        text_results = self.text_search(text, top_k=top_k[0], table=table, where=where, select=pre_select,
                                        nprobes=nprobes, refine_factor=refine_factor, output_format='pyarrow')

        tmp_table_name = uuid.uuid4().hex
        tmp_table = self.db.create_table(tmp_table_name, data=text_results)

        results = self.image_search(image_embedding, top_k=top_k[1], table=tmp_table, select=select, nprobes=nprobes,
                                    refine_factor=refine_factor, output_format=output_format)
        self.db.drop_table(tmp_table_name)

        return results
