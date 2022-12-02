package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.WHILE;

@Data
public class WhileStatement extends Statement {
    private Expression condition;
    private Statement body;

    public WhileStatement(Node prarent, Expression condition, Statement body) {
        super(prarent);
        this.condition = condition;
        this.body = body;

        this.condition.setPrarent(this);
        this.body.setPrarent(this);

        getChildrens().addAll(condition, body);
    }

    public static void parser(Node node) {
        if (node instanceof WhileStatement) return;
        DoWhileStatement.parser(node);
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(WHILE) && n instanceof ParametersExpression) {
                    if (b instanceof BlockStatement) {
                        //remove WhileNode and Parameters
                        a.getChildrens().removeAll(List.of(m, n));
                        //create WhileNode and set prarent, condition
                        WhileStatement statement = new WhileStatement(node, (Expression) n, (BlockStatement) b);
                        //remove WhileNode body
                        node.getChildrens().remove(b);
                        node.replace(a, statement);
                        list.remove(b);
                    } else {
                        //remove WhileNode and Parameters
                        a.getChildrens().removeAll(m, n);
                        //create BlockNode and set Childrens
                        BlockStatement block = new BlockStatement(null, a.getChildrens());
                        WhileStatement statement = new WhileStatement(node, (Expression) n, block);
                        //replace this node whit WhileNode
                        node.replace(a, statement);
                    }
                }
            });
        });
    }
}