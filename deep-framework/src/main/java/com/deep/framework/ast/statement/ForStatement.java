package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.lexer.TokenType;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.FOR;

@Data
public class ForStatement extends Statement {
    private Expression initializer;
    private Expression condition;
    private Expression update;
    private BlockStatement body;

    public ForStatement(Node prarent, Expression initializer, Expression condition, Expression update, BlockStatement body) {
        super(prarent);
        this.initializer = initializer;
        this.condition = condition;
        this.update = update;
        this.body = body;

        this.initializer.setPrarent(this);
        this.condition.setPrarent(this);
        this.update.setPrarent(this);
        this.body.setPrarent(this);

        getChildrens().addAll(initializer, condition, update, body);
    }

    public static void parser(Node node) {
        if (node instanceof ForStatement) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(FOR) && (n instanceof ParametersExpression o && !o.getChildrens().get(0).getChildrens().stream().filter(e -> Objects.nonNull(e.getTokenType()) && e.getTokenType().equals(TokenType.COLON)).findFirst().isPresent())) {
                    List<Node> nodes = n.getChildrens();
                    if (b instanceof BlockStatement) {
                        //create ForNode and set Prarent，Parameters
                        ForStatement statement = new ForStatement(node, (Expression) nodes.get(0), (Expression) nodes.get(1), (Expression) nodes.get(2), (BlockStatement) b);
                        //remove ForNode and Parameters
                        a.getChildrens().removeAll(m, n);
                        node.replace(a, statement);
                        node.getChildrens().remove(b);
                        list.remove(b);
                    } else {
                        //remove ForNode and Parameters
                        a.getChildrens().removeAll(m, n);
                        //create BlockNode and set Childrens
                        BlockStatement block = new BlockStatement(null, a.getChildrens());
                        //create ForNode and set Prarent , Parameters
                        ForStatement statement = new ForStatement(node, (Expression) nodes.get(0), (Expression) nodes.get(1), (Expression) nodes.get(2), block);
                        //replace this node whit ForNode
                        node.replace(a, statement);
                    }
                }
            });
        });
    }
}