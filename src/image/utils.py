def skill2step_annotations(annotations: list[dict]) -> list[dict]:
    """
    Convert skill annotations to step annotations.
    :param annotations: List of annotations.
    :return: List of annotations.
    """
    all_steps = []
    for anno in annotations:
        for i, step in enumerate(anno['annotation']):
            step_info = step['subclip']
            if i > 0:
                step_info['ref_frame'] = f'{step_info["id"]}.png'
            all_steps.append(step_info)

    return all_steps
