package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.*;

public class TypeParametersExpression extends Expression {
    private Expression expression;

    public TypeParametersExpression(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Name.parser(node);
        Stream.of(node.getChildrens()).reduce((list, m, n, o) -> {
            if (Objects.nonNull(m) && Objects.nonNull(n) && Objects.nonNull(o)) {
                if (m.equals(LT) && o.equals(GT)) {
                    TypeParametersExpression parameters = new TypeParametersExpression(node);
                    parameters.setExpression((Expression) n);
                    parameters.getChildrens().add(n);
                    node.replace(n, parameters);
                    node.getChildrens().removeAll(List.of(m, o));
                    list.clear();
                    parser(node);
                } else if (m.equals(LT) && o.equals(RSHIFT)) {
                    TypeParametersExpression parameters = new TypeParametersExpression(node);
                    parameters.setExpression((Expression) n);
                    parameters.getChildrens().add(n);
                    node.replace(n, parameters);
                    node.replace(o, GT.getToken());
                    node.getChildrens().remove(m);
                    list.clear();
                    parser(node);
                } else if (m.equals(LT) && o.equals(URSHIFT)) {
                    TypeParametersExpression parameters = new TypeParametersExpression(node);
                    parameters.setExpression((Expression) n);
                    parameters.getChildrens().add(n);
                    node.replace(n, parameters);
                    node.replace(o, RSHIFT.getToken());
                    node.getChildrens().remove(m);
                    list.clear();
                    parser(node);
                }
            }
        });
    }

    public void setExpression(Expression expression) {
        this.expression = expression;
    }
}