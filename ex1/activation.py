import numpy as np

def activation_function(name):
        if name == "step":
            def fn(state): 
                return np.heaviside(state, 1)
            return fn  
        if name == "sigmoid":
            def fn(state):
                return 1 / (1 + np.exp(-state))
            return fn
        if name == "sin":
            def fn(state):
                return np.clip(np.sin(state), -1, 1)
            return fn
        if name == "tanh":
            def fn(state):
                return np.tanh(state)
            return fn
        if name == "sign":
            def fn(state):
                return np.sign(state)
            return fn
        if name == "relu":
            def fn(state):
                return np.clip(state * (state > 0), -1, 1)
            return fn
        if name == "lrelu":
            def fn(state):
                return np.clip(np.where(state > 0, state, 0.01 * state), -1, 1)
            return fn
        else:
            raise NotImplementedError

def d_activation_funcion(name):
    if name == "step":
        def fn(state): 
            return 1
        return fn  
    if name == "sigmoid":
        def fn(state):
            return (1 / (1 + np.exp(-state))) * (1 - (1 / (1 + np.exp(-state))))
        return fn
    if name == "sin":
        def fn(state):
            return np.cos(state)
        return fn
    if name == "tanh":
        def fn(state):
            return 1.0 - np.tanh(state)**2
        return fn
    if name == "sign":
        def fn(state):
            return 1
        return fn
    if name == "sign":
        def fn(state):
            return 1
        return fn
    if name == "relu":
        def fn(state):
            return np.where(state > 0, 1.0, 0.0)
        return fn
    if name == "lrelu":
        def fn(state):
            return np.where(state > 0, 1.0, 0.01)
        return fn
    else:
        raise NotImplementedError