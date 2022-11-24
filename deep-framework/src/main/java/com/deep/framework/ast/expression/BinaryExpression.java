package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.lexer.TokenType;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.*;

@Data
public class BinaryExpression extends Expression {
    private Expression left;
    private Expression right;
    private TokenType operator;

    private static BinaryExpression expression;

    public BinaryExpression(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, m, n, o) -> {
            if (Objects.nonNull(m) && Objects.nonNull(n) && Objects.nonNull(o)) {
                if (Stream.of(MUL, DIV).contains(n.getTokenType())) {
                    expression = new BinaryExpression(node);
                    expression.getChildrens().addAll(List.of(m, o));

                    Expression a = (Expression) m;
                    Expression c = (Expression) o;

                    a.setPrarent(expression);
                    c.setPrarent(expression);

                    expression.setLeft(a);
                    expression.setOperator(n.getTokenType());
                    expression.setRight(c);

                    node.replace(m, expression);
                    node.getChildrens().removeAll(List.of(n, o));
                    list.clear();

                    parser(node);
                }
            }
        });

        Stream.of(node.getChildrens()).reduce((list, m, n, o) -> {
            if (Objects.nonNull(m) && Objects.nonNull(n) && Objects.nonNull(o)) {
                if (Stream.of(ADD, SUB).contains(n.getTokenType())) {
                    expression = new BinaryExpression(node);
                    expression.getChildrens().addAll(List.of(m, o));

                    Expression a = (Expression) m;
                    Expression c = (Expression) o;

                    a.setPrarent(expression);
                    c.setPrarent(expression);

                    expression.setLeft(a);
                    expression.setOperator(n.getTokenType());
                    expression.setRight(c);

                    node.replace(m, expression);
                    node.getChildrens().removeAll(List.of(n, o));
                    list.clear();

                    parser(node);
                }
            }
        });
    }

    public String toString() {
        return left.toString() + operator.toString() + right.toString();
    }
}
