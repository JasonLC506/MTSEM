

def simple_evaluate(
        model,
        data
):
    results = model.test(
        data_generator=data
    )
    return results
