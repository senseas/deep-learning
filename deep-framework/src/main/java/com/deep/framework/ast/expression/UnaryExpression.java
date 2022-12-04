package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.literal.NumericLiteral;

import java.sql.CallableStatement;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.*;

public class UnaryExpression extends Expression {
    private Expression expression;
    private TokenType operator;

    public UnaryExpression(Node prarent, Expression expression, TokenType operator) {
        super(prarent);
        this.operator = operator;
        this.expression = expression;
        this.expression.setPrarent(this);
        getChildrens().add(expression);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (Objects.nonNull(m) && Objects.nonNull(n)) {
                if (Stream.of(INC, DEC).contains(n.getTokenType())) {
                    UnaryExpression expression = new UnaryExpression(node, (Expression) m, n.getTokenType());
                    node.replace(m, expression);
                    node.getChildrens().remove(n);
                    list.clear();
                    parser(node);
                }
            }
        });

        Stream.of(node.getChildrens()).reduce((list, m, n, o) -> {
            if (Objects.nonNull(m) && Objects.nonNull(n)) {
                if (!(m instanceof BinaryExpression ||m instanceof CallableStatement|| m instanceof Name|| m instanceof NumericLiteral) && Stream.of(ADD, SUB, TILDE, BANG).contains(n.getTokenType())) {
                    UnaryExpression expression = new UnaryExpression(node, (Expression) o, n.getTokenType());
                    node.replace(n, expression);
                    node.getChildrens().remove(o);
                    list.clear();
                    parser(node);
                }else if (Stream.of(INC, DEC).contains(m.getTokenType())) {
                    UnaryExpression expression = new UnaryExpression(node, (Expression) n, m.getTokenType());
                    node.replace(m, expression);
                    node.getChildrens().remove(n);
                    list.clear();
                    parser(node);
                }
            }
        });
    }

}