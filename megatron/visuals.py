# This module credits heavy inspiration to a similar Keras module
# https://github.com/keras-team/keras/blob/master/keras/utils/vis_utils.py

import os
from .utils.generic import listify
from IPython.display import SVG
from .nodes import InputNode

# check for pydot
try:
    import pydot
except ImportError:
    pydot = None


def _check_pydot():
    """Raise errors if `pydot` or PipelineViz are not properly installed."""
    if pydot is None:
        raise ImportError('Failed to import `pydot`. Please install `pydot` in your '
                          'current environment.')
    try:
        pydot.Dot.create(pydot.Dot())
    except OSError:
        raise OSError('PipelineViz must be installed with its executables included in the $PATH.')


def pipeline_to_dot(pipeline, output_nodes, rankdir='TB'):
    """Convert a megatron Pipeline to dot format for visualization.

    Parameters
    ----------
    pipeline : megatron.Pipeline
        Feature pipeline defined as a pipeline.
    output_nodes : megatron.Node or list of megatron.Node
        The output nodes of the pipeline determine your feature-space. Include a list
        of all nodes which you would like to be included as features in the output.
    rankdir : str ['TB' or 'LR']
        Direction of pipeline to plot (top to bottom or left to right).


    Returns
    -------
    pydot.Dot
        Dot representation of the Pipeline.
    """

    _check_pydot()
    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    # build pipeline
    output_nodes = set(listify(output_nodes))
    path = pipeline._topsort(output_nodes)

    # add nodes
    for node in path:
        node_id = str(id(node))
        label = node.name
        # make input nodes green, output nodes blue
        if isinstance(node, InputNode):
            color = '#aeffad'
        elif node in output_nodes:
            color = '#b7e3ff'
        else:
            color = '#e8e8e8'
        pydot_node = pydot.Node(node_id, label=label, style='filled', fillcolor=color)
        dot.add_node(pydot_node)

        # create edges
        for input_node in reversed(node.inbound_nodes):
            input_node_id = str(id(input_node))
            dot.add_edge(pydot.Edge(input_node_id, node_id))

    return dot


def pipeline_imshow(pipeline, output_nodes, rankdir='TB'):
    """Create visualization of pipeline within Jupyter Notebook.

    Parameters
    ----------
    pipeline : megatron.Pipeline
        Feature pipeline defined as a pipeline.
    output_nodes : megatron.Node or list of megatron.Node
        The output nodes of the pipeline determine your feature-space. Include a list
        of all nodes which you would like to be included as features in the output.
    rankdir : str ['TB' or 'LR']
        Direction of pipeline to plot (top to bottom or left to right).

    Returns
    -------
    IPython.display.SVG
        Display of pipeline.
    """
    dot = pipeline_to_dot(pipeline, output_nodes, rankdir)
    return SVG(dot.create(prog='dot', format='svg'))


def pipeline_imsave(pipeline, output_nodes, save_path='pipeline.png', rankdir='TB'):
    """Save visualization of pipeline to an image file.

    Parameters
    ----------
    pipeline : megatron.Pipeline
        Feature pipeline defined as a pipeline.
    output_nodes : megatron.Node or list of megatron.Node
        The output nodes of the pipeline determine your feature-space. Include a list
        of all nodes which you would like to be included as features in the output.
    save_path : str
        Specify where to save the pipeline visualization.
    rankdir : str ['TB' or 'LR']
        Direction of pipeline to plot (top to bottom or left to right).
    """
    dot = pipeline_to_dot(pipeline, output_nodes, rankdir)
    _, extension = os.path.splitext(save_path)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(save_path, format=extension)
