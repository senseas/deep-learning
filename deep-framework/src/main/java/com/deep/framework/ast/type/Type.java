package com.deep.framework.ast.type;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;

public class Type extends Node {
    private Name name;
    private String modifier;

    public Type(Name name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name.toString();
    }
}
