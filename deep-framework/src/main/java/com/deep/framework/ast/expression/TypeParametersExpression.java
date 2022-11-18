package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.declaration.ClassOrInterfaceDeclaration;
import com.deep.framework.ast.declaration.MethodDeclaration;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.GT;
import static com.deep.framework.ast.lexer.TokenType.LT;

public class TypeParametersExpression extends Expression {
    public TypeParametersExpression(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (!(node instanceof MethodDeclaration || node instanceof ClassOrInterfaceDeclaration)) return;
        if (node.getChildrens().contains(LT)) {
            TypeParametersExpression m = null;
            for (Object a : List.copyOf(node.getChildrens())) {
                if (a.equals(GT)) {
                    node.remove(a);
                    node.getChildrens().removeAll(m.getChildrens());
                    return;
                } else if (a.equals(LT)) {
                    m = new TypeParametersExpression(node);
                    node.replace(a, m);
                } else if (Objects.nonNull(m)) {
                    m.getChildrens().add(a);
                }
            }
        }
    }
}