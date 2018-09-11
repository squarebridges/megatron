from ..core import Input


def from_dataframe(df, graph):
    '''
    Automatically create Input nodes for each column in a Pandas dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        the source of input data for your pipeline
    graph : megatron.Graph
        the pipeline you would like to connect to

    Usage
    -----
    Use this function to build a dictionary of Input nodes, connect Inputs to the
    pipeline by querying the dictionary by the column name of the corresponding dataframe.

    ```
    df_nodes = megatron.from_dataframe(df, G)
    X1 = megatron.transforms.SklearnTransformation(StandardScaler())(df_nodes['colname1'])
    X2 = megatron.transforms.SklearnTransformation(StandardScaler())(df_nodes['colname2'])
    ```

    Returns
    -------
    a dictionary with column names as keys and Input nodes as their values

    ```
    >>> from_dataframe(df, G)
     {colname1: Input(graph=G, name=colname1, input_shape=(1,)),
      colname2: Input(graph=G, name=colname2, input_shape=(1,)),
      ...}
    ```
    '''
    if graph.eager:
        out = {col: Input(graph=graph, name=col)(df[col].values) for col in df.columns}
    else:
        out = {col: Input(graph=graph, name=col) for col in df.columns}
    return out
