package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.GT;
import static com.deep.framework.ast.lexer.TokenType.LT;

public class TypeParametersExpression extends Expression {

    public TypeParametersExpression(Node prarent) {
        super(prarent);
    }

    private Expression expression;

    private static TypeParametersExpression typeParameters;

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((List list, Object m, Object n, Object o) -> {
            if (Objects.nonNull(m) && Objects.nonNull(n) && Objects.nonNull(o)) {
                if (m.equals(LT)) {
                    if (o.equals(GT)) {
                        typeParameters = new TypeParametersExpression(node);
                        typeParameters.setExpression((Expression) n);
                        typeParameters.getChildrens().add((Expression) n);
                        node.replace(n, typeParameters);
                        node.getChildrens().removeAll(List.of(m, o));
                        list.removeAll(List.of(n, o));
                        parser(node);
                    }
                }
            }
        });
    }

    public void setExpression(Expression expression) {
        this.expression = expression;
    }
}