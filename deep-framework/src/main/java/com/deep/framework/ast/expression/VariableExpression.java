package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.NodeList;
import com.deep.framework.ast.declaration.Declaration;
import com.deep.framework.ast.declaration.VariableDeclaration;
import lombok.Data;

import java.util.Objects;
import java.util.stream.Collectors;

@Data
public class VariableExpression extends Declaration {
    private NodeList<VariableDeclaration> variables;

    public VariableExpression(Node prarent, NodeList<VariableDeclaration> variables) {
        super(prarent);
        this.variables = variables;
        variables.forEach(a -> a.setPrarent(this));
        getChildrens().addAll(variables);
    }

    @Override
    public String toString() {
        return variables.stream().map(Objects::toString).collect(Collectors.joining(","));
    }
}