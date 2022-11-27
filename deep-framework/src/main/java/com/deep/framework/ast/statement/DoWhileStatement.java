package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.DO;
import static com.deep.framework.ast.lexer.TokenType.WHILE;

@Data
public class DoWhileStatement extends Statement {
    private List<Expression> initialization;
    private Expression compare;
    private List<Expression> update;
    private BlockStatement body;
    private ParametersExpression parameters;
    private static DoWhileStatement statement;

    public static void parser(Node node) {
        if (node instanceof ForStatement) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(DO) && b instanceof BlockStatement) {
                    //create ForNode and set Prarent，Parameters
                    b.setPrarent(statement);
                    statement = new DoWhileStatement();
                    statement.setPrarent(node);
                    statement.setBody((BlockStatement) b);
                    statement.getChildrens().add(b);

                    //remove ForNode and Parameters
                    a.getChildrens().remove(m);
                    node.replace(a, statement);
                    node.getChildrens().remove(b);
                    list.remove(b);
                } else if (m.equals(WHILE) && n instanceof ParametersExpression) {
                    n.setPrarent(statement);
                    statement.getChildrens().add(n);
                    statement.setParameters((ParametersExpression) n);
                }
            });
        });
    }

}