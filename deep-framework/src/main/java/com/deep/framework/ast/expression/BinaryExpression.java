package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.lexer.Token;
import com.deep.framework.ast.lexer.TokenType;

import java.util.List;
import java.util.Objects;


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
                if (Objects.nonNull(n.getTokenType()) && List.of(TokenType.MUL, TokenType.DIV).contains(n.getTokenType())) {
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
                    node.remove(n);
                    node.remove(o);
                    list.remove(n);
                    list.remove(o);
                    list.clear();

                    parser(node);
                }
            }
        });

        Stream.of(node.getChildrens()).reduce((list, m, n, o) -> {
            if (Objects.nonNull(m) && Objects.nonNull(n) && Objects.nonNull(o)) {
                if (Objects.nonNull(n.getTokenType()) && List.of(TokenType.ADD, TokenType.SUB).contains(n.getTokenType())) {
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
                    node.remove(n);
                    node.remove(o);
                    list.remove(n);
                    list.remove(o);
                    list.clear();

                    parser(node);
                }
            }
        });
    }

    public void setLeft(Expression left) {
        this.left = left;
    }

    public void setRight(Expression right) {
        this.right = right;
    }

    public void setOperator(TokenType operator) {
        this.operator = operator;
    }

    public static void setExpression(BinaryExpression expression) {
        BinaryExpression.expression = expression;
    }

    public String toString() {
        return left.toString() + operator.toString() + right.toString();
    }
}
