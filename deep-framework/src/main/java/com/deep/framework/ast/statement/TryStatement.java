package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.FINALLY;
import static com.deep.framework.ast.lexer.TokenType.TRY;

@Data
public class TryStatement extends Statement {
    private Expression condition;
    public ParametersExpression parameters;
    private BlockStatement tryBody;
    private Statement catchClause;
    private BlockStatement finallyBlock;
    private static TryStatement statement;

    public static void parser(Node node) {
        if (node instanceof TryStatement) return;
        Stream.of(node.getChildrens()).reduce((list, a, b, c) -> {
            Stream.of(a.getChildrens()).reduce((o, m, n) -> {
                if (m.equals(TRY) && n instanceof ParametersExpression) {
                    if (b instanceof BlockStatement) {
                        //create TryNode and set Prarent，Parameters
                        statement = new TryStatement();
                        statement.setPrarent(node);
                        statement.setParameters((ParametersExpression) n);
                        b.setPrarent(statement);
                        statement.setTryBody((BlockStatement) b);
                        statement.getChildrens().addAll(List.of(n, b));

                        if (c instanceof CatchClause) {
                            statement.setCatchClause((Statement) c);
                            node.getChildrens().remove(c);
                        }

                        //remove TryNode and Parameters
                        a.getChildrens().removeAll(List.of(m, n));
                        node.replace(a, statement);
                        node.getChildrens().remove(b);
                        list.remove(b);
                    } else {
                        throw new RuntimeException("TryNode error ".concat(node.toString()));
                    }
                } else if (m.equals(TRY)) {
                    if (b instanceof BlockStatement) {
                        //create TryNode and set Prarent，Parameters
                        statement = new TryStatement();
                        statement.setPrarent(node);
                        b.setPrarent(statement);
                        statement.setTryBody((BlockStatement) b);
                        statement.getChildrens().add(b);

                        if (c instanceof CatchClause) {
                            statement.setCatchClause((Statement) c);
                            node.getChildrens().remove(c);
                        }

                        //remove TryNode and Parameters
                        node.replace(a, statement);
                        node.getChildrens().remove(b);
                        list.remove(b);
                    } else {
                        throw new RuntimeException("TryNode error ".concat(node.toString()));
                    }
                } else if (m.equals(FINALLY)) {
                    statement.setFinallyBlock((BlockStatement) b);
                    node.getChildrens().remove(a);
                    node.getChildrens().remove(b);
                    list.remove(b);
                }
            });
        });
    }
}