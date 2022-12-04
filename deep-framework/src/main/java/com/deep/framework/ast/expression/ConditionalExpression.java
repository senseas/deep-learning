package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.NodeList;
import com.deep.framework.ast.Stream;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.COLON;
import static com.deep.framework.ast.lexer.TokenType.QUESTION;

@Data
public class ConditionalExpression extends Expression {
    private Expression condition;
    private Expression trueExpression;
    private Expression falseExpression;

    private static ConditionalExpression expression;

    public ConditionalExpression(Node prarent, Expression condition, Expression trueExpression, Expression falseExpression) {
        super(prarent);
        this.condition = condition;
        this.trueExpression = trueExpression;
        this.falseExpression = falseExpression;

        this.condition.setPrarent(this);
        this.trueExpression.setPrarent(this);
        this.falseExpression.setPrarent(this);

        getChildrens().addAll(condition, trueExpression, falseExpression);
    }

    public static void parser(Node node) {
        BinaryExpression.parser(node);
        Stream.of(node.getChildrens()).reduce((list, a, b, c, d) -> {
            if (Objects.nonNull(b) && b.equals(QUESTION) && Objects.nonNull(d) && d.equals(COLON)) {
                NodeList<Node> splita = node.split(QUESTION);
                NodeList<Node> splitb = splita.get(1).split(COLON);
                node.getChildrens().removeAll(List.of(b, d));
                //create SynchronizedNode and set Prarent，Parameters
                ConditionalExpression expression = new ConditionalExpression(node, (Expression) splita.get(0), (Expression) splitb.get(0), (Expression) splitb.get(1));
                //replace this node with SynchronizedNode
                node.replace(a, expression);
                node.getChildrens().removeAll(List.of(expression.getCondition(),expression.getTrueExpression(),expression.getFalseExpression()));
                list.clear();
                parser(node);
            }
        });
    }

}