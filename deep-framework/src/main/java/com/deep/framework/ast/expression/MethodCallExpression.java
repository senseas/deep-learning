package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.declaration.CallableDeclaration;
import com.deep.framework.ast.declaration.TypeDeclaration;
import lombok.Data;

@Data
public class MethodCallExpression extends Expression {
    private Expression arguments;
    private Expression name;

    public MethodCallExpression(Node prarent, Expression name, Expression arguments) {
        super(prarent);
        this.name = name;
        this.arguments = arguments;

        this.name.setPrarent(this);
        this.arguments.setPrarent(this);

        getChildrens().addAll(name, arguments);
    }

    public static void parser(Node node) {
        if (node instanceof MethodCallExpression) return;
        if (node.getPrarent() instanceof TypeDeclaration) return;
        Stream.of(node.getChildrens()).reduce2((list, a, b) -> {
            if (a instanceof CallableDeclaration c) {
                MethodCallExpression expression = new MethodCallExpression(node, c.getName(), c.getParameters());
                list.replace(a, expression);
            }
        });
    }

    @Override
    public String toString() {
        return name.toString().concat(arguments.toString());
    }
}