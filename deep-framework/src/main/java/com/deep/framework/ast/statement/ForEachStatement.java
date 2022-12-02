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
public class ForEachStatement extends Statement {
    private Expression variable;
    private Expression expression;
    private BlockStatement body;

    public ForEachStatement(Node prarent, Expression variable, Expression expression, BlockStatement body) {
        super(prarent);
        this.variable = variable;
        this.expression = expression;
        this.body = body;

        this.variable.setPrarent(this);
        this.expression.setPrarent(this);
        this.body.setPrarent(this);

        getChildrens().addAll(variable, expression, body);
    }

    public static void parser(Node node) {
        if (node instanceof ForStatement) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                List<Node> nodes = n.getChildrens();
                if (m.equals(FOR) && n instanceof ParametersExpression o && o.getChildrens().get(0).getChildrens().stream().filter(e -> Objects.nonNull(e.getTokenType()) && e.getTokenType().equals(TokenType.COLON)).findFirst().isPresent()) {
                    //create ForNode and set Prarent , Parameters
                    if (b instanceof BlockStatement) {
                        //remove ForNode and Parameters
                        a.getChildrens().removeAll(List.of(m, n));
                        ForEachStatement statement = new ForEachStatement(node, (Expression) nodes.get(0), (Expression) nodes.get(2), (BlockStatement) b);
                        node.replace(a, statement);
                        node.getChildrens().remove(b);
                        list.remove(b);
                    } else {
                        //remove ForNode and Parameters
                        a.getChildrens().removeAll(List.of(m, n));
                        //create BlockNode and set Childrens
                        BlockStatement block = new BlockStatement(null, a.getChildrens());
                        //create ForNode and set Prarent，Parameters
                        ForEachStatement statement = new ForEachStatement(node, (Expression) nodes.get(0), (Expression) nodes.get(2), block);
                        //replace this node whit ForNode
                        node.replace(a, statement);
                    }
                }
            });
        });
    }

}